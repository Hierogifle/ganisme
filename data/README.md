# Data

The data is **not redistributed** in this repository: the images belong to the
[Web Gallery of Art](https://www.wga.hu) and the catalogue was provided by the school.

## What is versioned

| File | Content |
|---|---|
| `datasets/summary.json` | filter funnel and dataset sizes |
| `datasets/portrait/metadata.csv` | the 3,832 portraits kept (source URL, train/test split, crop loss) |
| `datasets/landscape/metadata.csv` | the 3,016 landscapes kept |

## How to rebuild everything

1. Put the catalogue provided with the project brief at `data/art_catalog.xlsx`.
2. Run `notebooks/01_eda_art_catalog.ipynb` → `data/processed/paintings_catalog.csv`.
3. Download the images (about 50 minutes, 650 MB, resumable):
   `python src/scrape_images.py` → `data/raw/images/`.
4. Run `notebooks/02_image_visualization.ipynb` → `data/processed/image_stats.csv`.
5. Build the training sets: `python src/dataset.py` → `data/datasets/<genre>/images/`.

Expected layout once rebuilt:

```
data/
├── art_catalog.xlsx            raw catalogue (52,867 artworks)
├── raw/images/                 32,437 paintings, 400 px on the long side
├── processed/                  enriched catalogue, image statistics
└── datasets/
    ├── portrait/               128×160 crops, 3,448 train / 384 test
    └── landscape/              128×96 crops, 2,714 train / 302 test
```
