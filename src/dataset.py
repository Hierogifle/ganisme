"""Build the GAN training sets, one per genre.

Pipeline (each step is justified in notebooks 01 and 02):

1. **Global filters** — remove whatever is not a usable painting: images that were not downloaded,
   architecture photos ("Photo" technique), monochrome images (sculptures, engravings, black-and-white
   photos of destroyed artworks), near-duplicates (details, multiple views), panels cut out on a studio
   background, extreme formats (predellas, wings).
2. **Genre selection** — keep the images of the genre whose format is close to the target format of
   the genre (4:5 vertical portrait, 4:3 horizontal landscape), so that cropping removes little.
3. **Crop + resize** — crop to the target aspect ratio (at most 25 % of the area lost), then Lanczos
   resize (anti-aliasing) to the highest planned resolution.
4. **Train / test split by artwork** — 10 % of the artworks are set aside: never seen during
   training, they are used to measure the "floor" of the metrics and to detect memorisation. The split
   is done by group (same author + same base title) so that one artwork never ends up on both sides.

Outputs, for each genre:
    data/datasets/<genre>/images/<id>.png   images cropped at the highest resolution
    data/datasets/<genre>/metadata.csv      one row per image (source, split, crop loss…)
    data/datasets/summary.json              filter funnel and dataset sizes

Usage:
    python src/dataset.py                      # every configured genre
    python src/dataset.py --genres portrait    # a single genre
    python src/dataset.py --force              # rebuild even if the folder exists
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

# Thresholds of the global filters (values chosen and checked visually in notebook 02)
MONOCHROME_MAX = 0.02      # gap between RGB channels / brightness: below this, the image is monochrome
CORNER_STD_MAX = 0.03      # corners with little variation…
CORNER_LUM_MIN = 0.55      # …and bright: studio background around a tondo / a cut-out panel
ASPECT_RANGE = (0.5, 2.0)  # outside this range: predellas, wings, friezes


@dataclass(frozen=True)
class GenreConfig:
    """Parameters of one genre dataset.

    `size` is the HIGHEST stored resolution (width, height). Lower training resolutions (e.g. 64×80)
    are obtained by dividing by 2: they keep the same aspect ratio, which allows progressive training
    64 → 128 without rebuilding the dataset.
    """
    name: str
    types: tuple[str, ...]        # values of the TYPE column of the catalogue
    size: tuple[int, int]         # (width, height) in pixels
    center_y: float = 0.5         # vertical position of the crop (0 = top, 1 = bottom)
    max_crop_loss: float = 0.25   # largest share of the area that may be lost by cropping
    test_frac: float = 0.10

    @property
    def aspect(self) -> float:
        return self.size[0] / self.size[1]

    def resolutions(self) -> list[tuple[int, int]]:
        w, h = self.size
        return [(w // 2, h // 2), (w, h)]


GENRES = {
    # Portraits: 90 % are vertical, median aspect ratio ≈ 0.8 → 4:5 format.
    # Crop shifted slightly upwards (0.4): the mean image shows the face in the upper third.
    "portrait": GenreConfig("portrait", ("portrait",), size=(128, 160), center_y=0.4),
    # Landscapes: 83 % are horizontal, median aspect ratio ≈ 1.33 → 4:3 format.
    "landscape": GenreConfig("landscape", ("landscape",), size=(128, 96)),
}


# --------------------------------------------------------------------------- loading
def load_images_table() -> pd.DataFrame:
    """Enriched catalogue + scraping result + visual statistics, one row per painting."""
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
    """Visual statistics needed by the filters (same computation as notebook 02)."""
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


# --------------------------------------------------------------------------- filters
def global_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the global filters in order and return (kept images, funnel).

    The funnel tells how many images each filter removes: it is the table to show when justifying
    the final size of the dataset.
    """
    chroma_rel = df["CHROMA_SPREAD"] / df["BRIGHTNESS"].clip(lower=0.05)
    steps = [
        ("Paintings in the catalogue", pd.Series(True, index=df.index)),
        ("Image downloaded", df["status"].eq("ok")),
        ("Not an architecture photo", ~df["MEDIUM"].str.lower().eq("photo")),
        ("Not monochrome", chroma_rel >= MONOCHROME_MAX),
        ("Not a near-duplicate", ~df["IS_NEAR_DUP"].astype(bool)),
        ("No studio background", ~((df["CORNER_STD"] < CORNER_STD_MAX) & (df["CORNER_LUM"] > CORNER_LUM_MIN))),
        (f"Aspect ratio between {ASPECT_RANGE[0]} and {ASPECT_RANGE[1]}", df["IMG_ASPECT"].between(*ASPECT_RANGE)),
    ]
    keep = pd.Series(True, index=df.index)
    funnel = []
    for label, mask in steps:
        before = int(keep.sum())
        keep &= mask.fillna(False)
        funnel.append({"step": label, "remaining": int(keep.sum()), "removed": before - int(keep.sum())})
    funnel[0]["removed"] = 0
    return df[keep].copy(), pd.DataFrame(funnel)


def crop_loss(aspect: pd.Series | float, target: float):
    """Share of the area lost when an image of aspect ratio `aspect` is cropped to `target`."""
    return 1 - np.minimum(aspect / target, target / aspect)


def select_genre(df: pd.DataFrame, cfg: GenreConfig) -> tuple[pd.DataFrame, dict]:
    """Keep the images of the genre for which cropping to the target format removes at most `max_crop_loss`."""
    g = df[df["TYPE"].isin(cfg.types)].copy()
    g["CROP_LOSS"] = crop_loss(g["IMG_ASPECT"], cfg.aspect)
    kept = g[g["CROP_LOSS"] <= cfg.max_crop_loss].copy()
    info = {"genre_after_filters": len(g), "compatible_format": len(kept), "removed_by_format": len(g) - len(kept)}
    return kept, info


def split_by_artwork(df: pd.DataFrame, test_frac: float, seed: int = SEED) -> pd.Series:
    """Train/test draw at the artwork level (author + base title) to avoid leakage.

    If two rows describe the same artwork (e.g. two versions of one portrait), they always fall on the
    same side: otherwise the "test" set would contain images already seen during training.
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
    """Stable, readable identifier derived from the URL: '…/html/a/aachen/adonis.html' -> 'a__aachen__adonis'."""
    return url.split("/html/", 1)[1].rsplit(".", 1)[0].replace("/", "__")


def crop_and_resize(src: Path, size: tuple[int, int], center_y: float) -> Image.Image:
    """Crop to the aspect ratio of `size` (keeping the largest possible area), then resize.

    `ImageOps.fit` cuts out the largest window with the right aspect ratio, positioned according to
    `centering` (0.5 horizontally; `center_y` vertically), then resizes. Lanczos is an anti-aliasing
    filter: without it, downscaling would produce jagged edges and moiré that the GAN would learn to
    reproduce.
    """
    with Image.open(src) as im:
        return ImageOps.fit(im.convert("RGB"), size, Image.LANCZOS, centering=(0.5, center_y))


def build_genre(df_filtered: pd.DataFrame, cfg: GenreConfig, force: bool = False) -> dict:
    """Build data/datasets/<genre>/ and return a summary in numbers."""
    out = DATASETS / cfg.name
    img_dir = out / "images"
    kept, info = select_genre(df_filtered, cfg)
    kept["ID"] = kept["URL"].map(image_id)
    kept["SPLIT"] = split_by_artwork(kept, cfg.test_frac)
    kept["FILE"] = "images/" + kept["ID"] + ".png"

    if force or not (out / "metadata.csv").exists():
        img_dir.mkdir(parents=True, exist_ok=True)
        for old in img_dir.glob("*.png"):       # start again from a clean folder
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
            "median_crop_loss": round(float(kept["CROP_LOSS"].median()), 3),
            "resolutions": cfg.resolutions()}


def load_split(genre: str, split: str | None = "train", size: tuple[int, int] | None = None) -> np.ndarray:
    """Load the images of a genre into memory: uint8 array (N, H, W, 3).

    `size` loads a lower resolution (e.g. (64, 80)): downscaling uses the same Lanczos filter as the
    dataset, so that real and generated images are compared at the same resolution and with the same
    processing.
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
    summary = {"global_filters": funnel.to_dict("records"), "genres": {}}
    for g in args.genres:
        summary["genres"][g] = build_genre(filtered, GENRES[g], force=args.force)
        print(g, summary["genres"][g])
    DATASETS.mkdir(parents=True, exist_ok=True)
    (DATASETS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
