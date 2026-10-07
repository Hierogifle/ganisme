"""Construction des jeux d'entraînement du GAN, un par genre.

Pipeline (chaque étape est justifiée dans notebooks/01 et 02) :

1. **Filtres globaux** — on retire ce qui n'est pas une peinture exploitable :
   images non téléchargées, photos d'architecture (technique « Photo »), images monochromes
   (sculptures, gravures, photos N&B d'œuvres détruites), quasi-doublons (détails, vues multiples),
   panneaux détourés sur fond de studio, formats extrêmes (prédelles, volets).
2. **Sélection du genre** — on garde les images du genre dont le format est proche du format cible
   du genre (portrait 4:5 vertical, paysage 4:3 horizontal), pour que le recadrage coupe peu.
3. **Recadrage + redimensionnement** — recadrage au ratio cible (perte ≤ 25 % de la surface),
   puis redimensionnement Lanczos (anti-aliasing) à la résolution maximale prévue.
4. **Séparation train / test par œuvre** — 10 % des œuvres sont mises de côté : jamais vues à
   l'entraînement, elles servent à mesurer le « plancher » des métriques et à détecter la mémorisation.
   La séparation se fait par groupe (même auteur + même titre de base) pour qu'une même œuvre ne se
   retrouve pas des deux côtés.

Sorties, pour chaque genre :
    data/datasets/<genre>/images/<id>.png   images recadrées à la résolution maximale
    data/datasets/<genre>/metadata.csv      une ligne par image (origine, split, recadrage…)
    data/datasets/summary.json              entonnoir des filtres et effectifs

Usage :
    python src/dataset.py                      # tous les genres configurés
    python src/dataset.py --genres portrait    # un seul genre
    python src/dataset.py --force              # reconstruit même si le dossier existe
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "processed" / "paintings_catalog.csv"
MANIFEST = ROOT / "data" / "raw" / "images_manifest.csv"
IMAGE_STATS = ROOT / "data" / "processed" / "image_stats.csv"
DATASETS = ROOT / "data" / "datasets"
SEED = 42

# Seuils des filtres globaux (valeurs choisies et validées visuellement dans le notebook 02)
MONOCHROME_MAX = 0.02      # écart entre canaux RGB / luminosité : en dessous, image monochrome
CORNER_STD_MAX = 0.03      # coins de variabilité faible…
CORNER_LUM_MIN = 0.55      # …et clairs : fond de studio autour d'un tondo / panneau détouré
ASPECT_RANGE = (0.5, 2.0)  # hors de cet intervalle : prédelles, volets, frises


@dataclass(frozen=True)
class GenreConfig:
    """Paramètres d'un jeu de données de genre.

    `size` est la résolution MAXIMALE stockée (largeur, hauteur). Les résolutions d'entraînement
    plus basses (ex. 64×80) sont obtenues en divisant par 2 : elles gardent le même ratio, ce qui
    permet un entraînement progressif 64 → 128 sans refaire le dataset.
    """
    name: str
    types: tuple[str, ...]        # valeurs de la colonne TYPE du catalogue
    size: tuple[int, int]         # (largeur, hauteur) en pixels
    center_y: float = 0.5         # position verticale du recadrage (0 = haut, 1 = bas)
    max_crop_loss: float = 0.25   # part maximale de la surface perdue au recadrage
    test_frac: float = 0.10

    @property
    def aspect(self) -> float:
        return self.size[0] / self.size[1]

    def resolutions(self) -> list[tuple[int, int]]:
        w, h = self.size
        return [(w // 2, h // 2), (w, h)]


GENRES = {
    # Portraits : 90 % sont verticaux, ratio médian ≈ 0,8 → format 4:5.
    # Recadrage légèrement remonté (0,4) : l'image moyenne montre le visage dans le tiers supérieur.
    "portrait": GenreConfig("portrait", ("portrait",), size=(128, 160), center_y=0.4),
    # Paysages : 83 % sont horizontaux, ratio médian ≈ 1,33 → format 4:3.
    "landscape": GenreConfig("landscape", ("landscape",), size=(128, 96)),
}


# --------------------------------------------------------------------------- chargement
def load_images_table() -> pd.DataFrame:
    """Catalogue enrichi + résultat du scraping + statistiques visuelles, une ligne par peinture."""
    cat = pd.read_csv(CATALOG)
    man = pd.read_csv(MANIFEST).drop_duplicates("url", keep="last")
    df = cat.merge(man[["url", "status", "local_path", "width", "height", "mode"]],
                   left_on="URL", right_on="url", how="left").drop(columns="url")
    ok = df["status"].eq("ok")
    stats = IMAGE_STATS if IMAGE_STATS.exists() else None
    if stats is None:
        compute_image_stats(df.loc[ok]).to_csv(IMAGE_STATS, index=False)
    df = df.merge(pd.read_csv(IMAGE_STATS), on="URL", how="left")
    df["IMG_ASPECT"] = df["width"] / df["height"]
    return df


def compute_image_stats(df: pd.DataFrame, res: int = 64) -> pd.DataFrame:
    """Statistiques visuelles nécessaires aux filtres (même calcul que le notebook 02)."""
    luma = np.array([0.299, 0.587, 0.114], dtype=np.float32)

    def feats(row):
        with Image.open(ROOT / row["local_path"]) as im:
            im.draft("RGB", (res * 2, res * 2))
            a = np.asarray(im.convert("RGB").resize((res, res), Image.BILINEAR), dtype=np.float32) / 255
        k = 6
        corners = np.concatenate([a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]).reshape(-1, 3)
        return {"URL": row["URL"], "BRIGHTNESS": float((a @ luma).mean()),
                "CHROMA_SPREAD": float((np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])).mean() / 2),
                "CORNER_STD": float(corners.std(axis=0).mean()), "CORNER_LUM": float((corners @ luma).mean())}

    with ThreadPoolExecutor(8) as pool:
        return pd.DataFrame(pool.map(feats, df.to_dict("records")))


# --------------------------------------------------------------------------- filtres
def global_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applique les filtres globaux dans l'ordre et renvoie (images gardées, entonnoir).

    L'entonnoir (funnel) indique combien d'images chaque filtre retire : c'est le tableau à montrer
    pour justifier la taille finale du dataset.
    """
    chroma_rel = df["CHROMA_SPREAD"] / df["BRIGHTNESS"].clip(lower=0.05)
    steps = [
        ("Peintures au catalogue", pd.Series(True, index=df.index)),
        ("Image téléchargée", df["status"].eq("ok")),
        ("Pas une photo d'architecture", ~df["MEDIUM"].str.lower().eq("photo")),
        ("Pas monochrome", chroma_rel >= MONOCHROME_MAX),
        ("Pas un quasi-doublon", ~df["IS_NEAR_DUP"].astype(bool)),
        ("Pas de fond de studio", ~((df["CORNER_STD"] < CORNER_STD_MAX) & (df["CORNER_LUM"] > CORNER_LUM_MIN))),
        (f"Ratio entre {ASPECT_RANGE[0]} et {ASPECT_RANGE[1]}", df["IMG_ASPECT"].between(*ASPECT_RANGE)),
    ]
    keep = pd.Series(True, index=df.index)
    funnel = []
    for label, mask in steps:
        before = int(keep.sum())
        keep &= mask.fillna(False)
        funnel.append({"étape": label, "restantes": int(keep.sum()), "retirées": before - int(keep.sum())})
    funnel[0]["retirées"] = 0
    return df[keep].copy(), pd.DataFrame(funnel)


def crop_loss(aspect: pd.Series | float, target: float):
    """Part de la surface perdue quand on recadre une image de ratio `aspect` au ratio `target`."""
    return 1 - np.minimum(aspect / target, target / aspect)


def select_genre(df: pd.DataFrame, cfg: GenreConfig) -> tuple[pd.DataFrame, dict]:
    """Garde les images du genre dont le recadrage au format cible coupe au plus `max_crop_loss`."""
    g = df[df["TYPE"].isin(cfg.types)].copy()
    g["CROP_LOSS"] = crop_loss(g["IMG_ASPECT"], cfg.aspect)
    kept = g[g["CROP_LOSS"] <= cfg.max_crop_loss].copy()
    info = {"genre_après_filtres": len(g), "format_compatible": len(kept), "retirées_format": len(g) - len(kept)}
    return kept, info


def split_by_artwork(df: pd.DataFrame, test_frac: float, seed: int = SEED) -> pd.Series:
    """Tirage train/test au niveau de l'œuvre (auteur + titre de base) pour éviter les fuites.

    Si deux lignes décrivent la même œuvre (ex. deux versions d'un même portrait), elles tombent
    toujours du même côté : sinon le « test » contiendrait des images déjà vues à l'entraînement.
    """
    groups = df["AUTHOR"] + " | " + df["TITLE_BASE"].fillna(df["TITLE"])
    uniq = groups.drop_duplicates().sample(frac=1, random_state=seed).tolist()
    sizes = groups.value_counts()
    target, test_groups, n = test_frac * len(df), set(), 0
    for grp in uniq:
        if n >= target:
            break
        test_groups.add(grp)
        n += sizes[grp]
    return np.where(groups.isin(test_groups), "test", "train")


# --------------------------------------------------------------------------- export
def image_id(url: str) -> str:
    """Identifiant stable et lisible, dérivé de l'URL : '…/html/a/aachen/adonis.html' -> 'a__aachen__adonis'."""
    return url.split("/html/", 1)[1].rsplit(".", 1)[0].replace("/", "__")


def crop_and_resize(src: Path, size: tuple[int, int], center_y: float) -> Image.Image:
    """Recadre au ratio de `size` (en gardant la plus grande zone possible) puis redimensionne.

    `ImageOps.fit` découpe la plus grande fenêtre au bon ratio, positionnée selon `centering`
    (0,5 horizontalement ; `center_y` verticalement), puis redimensionne. Lanczos est un filtre
    anti-aliasing : sans lui, la réduction ferait apparaître du crénelage et du moiré que le GAN
    apprendrait à reproduire.
    """
    with Image.open(src) as im:
        return ImageOps.fit(im.convert("RGB"), size, Image.LANCZOS, centering=(0.5, center_y))


def build_genre(df_filtered: pd.DataFrame, cfg: GenreConfig, force: bool = False) -> dict:
    """Construit data/datasets/<genre>/ et renvoie un résumé chiffré."""
    out = DATASETS / cfg.name
    img_dir = out / "images"
    kept, info = select_genre(df_filtered, cfg)
    kept["ID"] = kept["URL"].map(image_id)
    kept["SPLIT"] = split_by_artwork(kept, cfg.test_frac)
    kept["FILE"] = "images/" + kept["ID"] + ".png"

    if force or not (out / "metadata.csv").exists():
        img_dir.mkdir(parents=True, exist_ok=True)
        for old in img_dir.glob("*.png"):       # repart d'un dossier propre
            old.unlink()

        def export(row):
            crop_and_resize(ROOT / row["local_path"], cfg.size, cfg.center_y).save(out / row["FILE"], optimize=True)

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(export, kept.to_dict("records")))
        cols = ["ID", "FILE", "SPLIT", "URL", "AUTHOR", "TITLE", "YEAR", "CENTURY", "SCHOOL", "TYPE",
                "MEDIUM_FAMILY", "width", "height", "IMG_ASPECT", "CROP_LOSS", "local_path"]
        kept[cols].rename(columns={"width": "SRC_WIDTH", "height": "SRC_HEIGHT", "local_path": "SRC_PATH"}) \
            .to_csv(out / "metadata.csv", index=False)

    counts = kept["SPLIT"].value_counts()
    return {**asdict(cfg), **info, "train": int(counts.get("train", 0)), "test": int(counts.get("test", 0)),
            "perte_recadrage_médiane": round(float(kept["CROP_LOSS"].median()), 3),
            "résolutions": cfg.resolutions()}


def load_split(genre: str, split: str | None = "train", size: tuple[int, int] | None = None) -> np.ndarray:
    """Charge les images d'un genre en mémoire : tableau uint8 (N, H, W, 3).

    `size` permet de charger une résolution plus basse (ex. (64, 80)) : la réduction utilise le même
    filtre Lanczos que le dataset, de sorte que les images réelles et générées sont comparées à la même
    résolution et avec le même traitement.
    """
    meta = pd.read_csv(DATASETS / genre / "metadata.csv")
    if split is not None:
        meta = meta[meta["SPLIT"] == split]

    def read(f):
        with Image.open(DATASETS / genre / f) as im:
            im = im.convert("RGB")
            return np.asarray(im if size is None or im.size == tuple(size) else im.resize(size, Image.LANCZOS))

    with ThreadPoolExecutor(8) as pool:
        return np.stack(list(pool.map(read, meta["FILE"])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--genres", nargs="+", default=list(GENRES), choices=list(GENRES))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    filtered, funnel = global_filters(load_images_table())
    print(funnel.to_string(index=False))
    summary = {"filtres_globaux": funnel.to_dict("records"), "genres": {}}
    for g in args.genres:
        summary["genres"][g] = build_genre(filtered, GENRES[g], force=args.force)
        print(g, summary["genres"][g])
    DATASETS.mkdir(parents=True, exist_ok=True)
    (DATASETS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
