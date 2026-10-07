"""Scraping des images de peintures de la Web Gallery of Art (www.wga.hu).

Lit le catalogue enrichi produit par le notebook d'EDA (data/processed/paintings_catalog.csv),
télécharge l'image de chaque peinture et tient un manifeste CSV de suivi.

Stratégie de résolution de l'URL de l'image :
  1. règle déduite de l'arborescence du site : /html/a/b/c.html -> /<taille>/a/b/c.jpg
     (validée sur un échantillon de pages : 40/40) ;
  2. en cas d'échec (404, réponse non-image), on scrape la page HTML de l'œuvre
     et on extrait le lien /art/... qu'elle contient.

Le script est reprenable : une image déjà présente et valide n'est jamais retéléchargée,
et chaque résultat est écrit immédiatement dans le manifeste. On peut l'interrompre (Ctrl+C)
et le relancer à tout moment.

Exemples :
    python src/scrape_images.py                    # toutes les peintures, taille 'detail' (400 px)
    python src/scrape_images.py --limit 50         # test rapide
    python src/scrape_images.py --size art         # pleine résolution (~10x plus lourd)
    python src/scrape_images.py --only-unique      # sans les détails / quasi-doublons
    python src/scrape_images.py --retry-failed     # retente uniquement les échecs précédents
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://www.wga.hu"
USER_AGENT = "GANisme-student-project/1.0 (La Plateforme, educational use)"
SIZES = {"detail": "detail", "art": "art", "thumb": "detail_s"}
MIN_SIDE = 32  # une image plus petite est considérée comme invalide

MANIFEST_FIELDS = ["url", "image_url", "local_path", "status", "http_status", "source",
                   "width", "height", "mode", "bytes", "attempts", "error", "timestamp"]
RE_ART_LINK = re.compile(r'href="(/art/[^"]+?\.(?:jpe?g|png|gif))"', re.I)

log = logging.getLogger("scraper")
_local = threading.local()


def session() -> requests.Session:
    # Une session HTTP par thread (requests.Session n'est pas garanti thread-safe).
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
        s.mount("https://", adapter)
        _local.session = s
    return _local.session


class NotFound(Exception):
    """Ressource absente (404 ou réponse qui n'est pas une image) : inutile de réessayer."""


def http_get(url: str, timeout: tuple[float, float], retries: int, attempts: list[int]) -> requests.Response:
    """GET avec backoff exponentiel sur les erreurs transitoires (timeouts, 429, 5xx)."""
    for attempt in range(retries + 1):
        attempts[0] += 1
        try:
            resp = session().get(url, timeout=timeout)
            if resp.status_code in (404, 410):
                raise NotFound(f"{resp.status_code} {url}")
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(exc.response, "status_code", None) if isinstance(exc, requests.HTTPError) else None
            if status is not None and status < 500 and status != 429:
                raise NotFound(f"HTTP {status} {url}") from exc   # 4xx : définitif
            if attempt == retries:
                raise
            wait = 2 ** attempt * 2                                  # 2, 4, 8, 16 s…
            retry_after = exc.response.headers.get("Retry-After") if status else None
            if retry_after and retry_after.isdigit():
                wait = max(wait, int(retry_after))
            wait += random.uniform(0, 1)                             # jitter : désynchronise les threads
            log.info("retry %d/%d dans %.1fs (%s) %s", attempt + 1, retries, wait, type(exc).__name__, url)
            time.sleep(wait)
    raise RuntimeError("inatteignable")


def image_from_response(resp: requests.Response) -> Image.Image:
    ctype = resp.headers.get("content-type", "")
    if not ctype.startswith("image"):
        raise NotFound(f"réponse non-image ({ctype})")
    img = Image.open(io.BytesIO(resp.content))
    img.load()  # force le décodage complet : détecte les fichiers tronqués
    if min(img.size) < MIN_SIDE:
        raise NotFound(f"image trop petite {img.size}")
    return img


def resolve_from_page(page_url: str, size_dir: str, timeout, retries, attempts) -> str:
    """Scrape la page HTML de l'œuvre pour y trouver le lien vers l'image."""
    html = http_get(page_url, timeout, retries, attempts).content.decode("iso-8859-1", errors="replace")
    m = RE_ART_LINK.search(html)
    if not m:
        raise NotFound("aucun lien /art/ dans la page")
    return BASE_URL + m.group(1).replace("/art/", f"/{size_dir}/", 1)


def local_path_for(page_url: str, out_dir: Path) -> Path:
    # Miroir de l'arborescence du site : unique, puisque l'URL est une clé primaire (cf. EDA).
    rel = page_url.split("/html/", 1)[1].rsplit(".", 1)[0] + ".jpg"
    return out_dir / rel


def is_valid_file(path: Path) -> tuple[int, int, str] | None:
    try:
        with Image.open(path) as img:
            img.load()
            if min(img.size) >= MIN_SIDE:
                return img.size[0], img.size[1], img.mode
    except Exception:
        pass
    return None


def download_one(row: dict, args, out_dir: Path) -> dict:
    page_url = row["URL"]
    size_dir = SIZES[args.size]
    dest = local_path_for(page_url, out_dir)
    rec = {"url": page_url, "local_path": dest.relative_to(ROOT).as_posix(), "attempts": 0,
           "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    attempts = [0]
    timeout = (10, args.timeout)

    # Image déjà présente sur disque et lisible : rien à faire.
    if dest.exists() and (info := is_valid_file(dest)):
        rec.update(status="ok", source="cache", width=info[0], height=info[1], mode=info[2],
                   bytes=dest.stat().st_size, image_url=row["IMAGE_URL"].replace("/art/", f"/{size_dir}/", 1))
        return rec

    candidates = [("rule", row["IMAGE_URL"].replace("/art/", f"/{size_dir}/", 1))]
    last_error, http_status = None, None
    for source, img_url in candidates + [("page", None)]:
        try:
            if source == "page":
                img_url = resolve_from_page(page_url, size_dir, timeout, args.retries, attempts)
                if img_url == candidates[0][1]:
                    raise NotFound("la page pointe vers la même image introuvable")
            resp = http_get(img_url, timeout, args.retries, attempts)
            img = image_from_response(resp)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(resp.content)  # octets d'origine : pas de ré-encodage JPEG
            tmp.replace(dest)              # écriture atomique : jamais de fichier à moitié écrit
            rec.update(status="ok", source=source, image_url=img_url, http_status=resp.status_code,
                       width=img.size[0], height=img.size[1], mode=img.mode, bytes=len(resp.content),
                       attempts=attempts[0])
            return rec
        except NotFound as exc:
            last_error, http_status = str(exc), 404
        except Exception as exc:  # erreurs réseau persistantes après retries
            last_error = f"{type(exc).__name__}: {exc}"[:300]
            http_status = getattr(getattr(exc, "response", None), "status_code", None)
            break  # inutile de scraper la page si le serveur ne répond pas
        finally:
            time.sleep(args.delay)  # politesse : pause entre deux requêtes d'un même thread

    rec.update(status="not_found" if http_status == 404 else "error", error=last_error,
               http_status=http_status, image_url=candidates[0][1], attempts=attempts[0])
    return rec


def load_manifest(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str)
    return pd.DataFrame(columns=MANIFEST_FIELDS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=ROOT / "data" / "processed" / "paintings_catalog.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw" / "images")
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "raw" / "images_manifest.csv")
    p.add_argument("--size", choices=list(SIZES), default="detail",
                   help="detail = 400 px (défaut, suffisant pour un GAN) ; art = pleine résolution ; thumb = vignette")
    p.add_argument("--workers", type=int, default=4, help="téléchargements simultanés (rester modeste)")
    p.add_argument("--delay", type=float, default=0.25, help="pause (s) entre deux requêtes d'un même thread")
    p.add_argument("--timeout", type=float, default=60, help="timeout de lecture (s)")
    p.add_argument("--retries", type=int, default=4)
    p.add_argument("--limit", type=int, help="ne traiter que les N premières peintures (test)")
    p.add_argument("--only-unique", action="store_true", help="ignorer les détails et quasi-doublons (IS_NEAR_DUP)")
    p.add_argument("--retry-failed", action="store_true", help="ne traiter que les échecs du manifeste")
    args = p.parse_args()

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.manifest.with_name("scraping.log"), encoding="utf-8")])

    cat = pd.read_csv(args.catalog)
    if args.only_unique:
        cat = cat[~cat["IS_NEAR_DUP"]]
    manifest = load_manifest(args.manifest)
    done_ok = set(manifest.loc[manifest["status"] == "ok", "url"])
    if args.retry_failed:
        failed = set(manifest.loc[manifest["status"] != "ok", "url"]) - done_ok
        todo = cat[cat["URL"].isin(failed)]
    else:
        todo = cat[~cat["URL"].isin(done_ok)]
    if args.limit:
        todo = todo.head(args.limit)

    msg = (f"{len(cat):,} peintures au catalogue | déjà OK : {len(done_ok):,} | à traiter : {len(todo):,} "
           f"| taille={args.size} workers={args.workers}").replace(",", " ")
    print(msg)
    log.info(msg)
    if todo.empty:
        return

    counts = {"ok": 0, "not_found": 0, "error": 0}
    lock = threading.Lock()
    write_header = not args.manifest.exists() or args.manifest.stat().st_size == 0
    with open(args.manifest, "a", newline="", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        futures = [pool.submit(download_one, r, args, args.out_dir) for r in todo.to_dict("records")]
        bar = tqdm(total=len(futures), unit="img", smoothing=0.05)
        try:
            for fut in as_completed(futures):
                rec = fut.result()
                with lock:
                    writer.writerow(rec)
                    fh.flush()
                counts[rec["status"]] += 1
                if rec["status"] != "ok":
                    log.warning("%s %s %s", rec["status"], rec["url"], rec.get("error"))
                bar.update()
                bar.set_postfix(ok=counts["ok"], nf=counts["not_found"], err=counts["error"])
        except KeyboardInterrupt:
            print("\nInterruption : annulation des tâches en attente (le manifeste est à jour, relancer pour reprendre).")
            for f in futures:
                f.cancel()
            raise
        finally:
            bar.close()

    msg = f"Terminé : {counts}"
    print(msg)
    log.info(msg)


if __name__ == "__main__":
    main()
