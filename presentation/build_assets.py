"""Build the images and the data file used by the interactive presentation (presentation/index.html).

Everything comes from files that already exist in the project: exported generators (models/), training
logs (runs/), result tables (reports/), figures (reports/figures/) and the portrait dataset. Nothing is
trained here: the generators are only used to produce images.

    python presentation/build_assets.py

Outputs:
    presentation/assets/...     images (characters, quiz, easel, morphing, figures)
    presentation/js/data.js     numbers used by the interactive charts (window.GANISME)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
import dataset as ds  # noqa: E402
import generate as gn  # noqa: E402

ASSETS = HERE / "assets"
FIGURES = ROOT / "reports" / "figures"
RUNS = ROOT / "runs"

# Images chosen by eye on contact sheets (see presentation/_work/): (seed, index in a batch of 60)
CAST = {"curator": (12, 36), "forger": (12, 22), "inspector": (12, 13)}
QUIZ_FAKES = [(11, 14), (11, 43), (12, 47), (11, 45)]
HUNG_PAINTING = (11, 30)                       # an ordinary, unflattering sample: the painting of the final twist
QUIZ_REALS = [3, 18, 38, 44]                   # rows of a fixed draw of 48 test portraits (random_state=3)
MORPH_PAIRS = [(43, 26), (24, 37)]             # pairs of single-image seeds
MORPH_STEPS = 16
EASEL_TILES = [(0, 2), (0, 3), (1, 5), (4, 7), (6, 7), (7, 2)]     # (row, column) in the 8×8 fixed-noise grid
LANDSCAPE_PICKS = [0, 1, 2, 4, 9, 12, 15, 25, 28, 29, 32, 36]       # in a batch of 40, seed 11

FIGURE_FILES = [
    "01_form_distribution", "04_paintings_type", "05_paintings_school", "06_paintings_time", "07_type_by_century",
    "18_cramers_v", "11_dimensions", "img_05_details_giotto", "img_08_photo", "img_07_fresco_insitu",
    "img_16_mono_border", "img_19_resize_strategies", "ds_02_aspect_selection", "ds_03_crop_examples",
    "ds_04_final_samples", "confirm_04_nearest_neighbors", "diffaugment_examples", "confirm_03_final_vs_real",
    "confirm_05_interpolation", "p128_02_fair_comparison", "p128_03_samples", "p128_04_nearest_neighbors",
    "landscape_03_fair_comparison", "landscape_04_samples", "landscape_06_nearest_neighbors", "optuna_03_effects",
    "img_11_color_by_century", "img_13_mean_by_type", "img_02_by_type",
]


# --------------------------------------------------------------------------- helpers
def save(img: Image.Image, path: Path, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, quality=quality, optimize=True)


def upscale(arr: np.ndarray, factor: int) -> Image.Image:
    im = Image.fromarray(arr)
    return im.resize((im.width * factor, im.height * factor), Image.LANCZOS)


def strip(images: list[np.ndarray] | np.ndarray, factor: int, gap: int = 6, bg=(24, 22, 20)) -> Image.Image:
    """Images side by side on a dark ground."""
    tiles = [upscale(x, factor) for x in images]
    w, h = tiles[0].size
    canvas = Image.new("RGB", (len(tiles) * w + (len(tiles) - 1) * gap, h), bg)
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * (w + gap), 0))
    return canvas


def grid_tiles(path: Path, tile_w: int, tile_h: int, pad: int = 2) -> np.ndarray:
    """Cut a grid saved by torchvision (padding of `pad` pixels) into tiles: array [rows, cols, H, W, 3]."""
    g = np.asarray(Image.open(path).convert("RGB"))
    rows, cols = (g.shape[0] - pad) // (tile_h + pad), (g.shape[1] - pad) // (tile_w + pad)
    return np.stack([np.stack([g[pad + r * (tile_h + pad): pad + r * (tile_h + pad) + tile_h,
                                 pad + c * (tile_w + pad): pad + c * (tile_w + pad) + tile_w]
                               for c in range(cols)]) for r in range(rows)])


def r(x, nd: int = 3):
    return None if pd.isna(x) else round(float(x), nd)


# --------------------------------------------------------------------------- generated images
def build_generated() -> dict:
    out = {}
    G, _ = gn.load_generator_file(ROOT / "models" / "portrait_128.pt", "cpu")
    batches = {seed: gn.sample(G, 60, seed=seed) for seed in (11, 12)}

    for name, (seed, i) in CAST.items():
        save(upscale(batches[seed][i], 3), ASSETS / "cast" / f"{name}.jpg", 92)
    seed, i = HUNG_PAINTING
    save(upscale(batches[seed][i], 4), ASSETS / "cast" / "hung_painting.jpg", 92)

    # Quiz: four generated portraits and four real ones, in a fixed shuffled order
    meta = pd.read_csv(ROOT / "data" / "datasets" / "portrait" / "metadata.csv")
    test = meta[meta["SPLIT"] == "test"].sample(48, random_state=3).reset_index(drop=True)
    items = [{"real": False, "img": batches[s][i], "caption": "Généré par le modèle 128×160"} for s, i in QUIZ_FAKES]
    for k in QUIZ_REALS:
        row = test.loc[k]
        img = np.asarray(Image.open(ROOT / "data" / "datasets" / "portrait" / row["FILE"]).convert("RGB"))
        year = "" if pd.isna(row["YEAR"]) else f", {int(row['YEAR'])}"
        items.append({"real": True, "img": img, "caption": f"{row['AUTHOR'].title()}{year}"})
    order = np.random.default_rng(7).permutation(len(items))
    out["quiz"] = []
    for n, k in enumerate(order):
        save(upscale(items[k]["img"], 2), ASSETS / "quiz" / f"q{n}.jpg", 92)
        out["quiz"].append({"file": f"assets/quiz/q{n}.jpg", "real": items[k]["real"], "caption": items[k]["caption"]})

    # Uncurated samples (seed 2024, the one used in the README) and morphing sequences
    save(gn.to_grid(gn.sample(G, 16, seed=2024), ncol=8, scale=2), ASSETS / "samples" / "portraits_128.jpg")
    out["morph"] = []
    for n, (a, b) in enumerate(MORPH_PAIRS):
        frames = gn.interpolate(G, a, b, MORPH_STEPS)
        for k, f in enumerate(frames):
            save(upscale(f, 3), ASSETS / "morph" / f"m{n}_{k:02d}.jpg", 90)
        out["morph"].append({"prefix": f"assets/morph/m{n}_", "steps": MORPH_STEPS, "seeds": [a, b]})

    G, _ = gn.load_generator_file(ROOT / "models" / "landscape_128.pt", "cpu")
    land = gn.sample(G, 40, seed=11)[LANDSCAPE_PICKS]
    save(gn.to_grid(land, ncol=6, scale=2), ASSETS / "samples" / "landscapes_128.jpg")
    save(gn.to_grid(gn.sample(G, 12, seed=2024), ncol=6, scale=2), ASSETS / "samples" / "landscapes_128_uncurated.jpg")
    frames = gn.interpolate(G, 0, 12, MORPH_STEPS)
    for k, f in enumerate(frames):
        save(upscale(f, 3), ASSETS / "morph" / f"l0_{k:02d}.jpg", 90)
    out["morph"].append({"prefix": "assets/morph/l0_", "steps": MORPH_STEPS, "seeds": [0, 12]})
    return out


# --------------------------------------------------------------------------- training logs
def build_easel() -> dict:
    """The same six fixed-noise portraits at every evaluation of the final 64×80 model."""
    run = RUNS / "optuna26_portrait_64_s1"
    ev = pd.read_csv(run / "eval.csv").set_index("epoch")
    frames = []
    for path in sorted((run / "samples").glob("epoch_*.png")):
        epoch = int(path.stem.split("_")[1])
        tiles = grid_tiles(path, 64, 80)
        save(strip([tiles[rr, cc] for rr, cc in EASEL_TILES], 3), ASSETS / "easel" / f"e{epoch:04d}.jpg", 88)
        frames.append({"epoch": epoch, "file": f"assets/easel/e{epoch:04d}.jpg",
                       "fid": r(ev.loc[epoch, "FID"], 1) if epoch in ev.index else None})
    return {"frames": frames}


def build_attempts() -> dict:
    """One row of fixed-noise samples for each hand-tuned model, at its best epoch (and at the collapse)."""
    picks = {"model1": ("dcgan_portrait_64", 400), "model2": ("dcgan_diffaug_portrait_64", 450),
             "model2_collapse": ("dcgan_diffaug_portrait_64", 500), "model3": ("sngan_diffaug_portrait_64", 500),
             "final": ("optuna26_portrait_64_s1", 600)}
    out = {}
    for key, (run, epoch) in picks.items():
        tiles = grid_tiles(RUNS / run / "samples" / f"epoch_{epoch:04d}.png", 64, 80)
        save(strip(list(tiles[0]) + list(tiles[1][:2]), 2), ASSETS / "attempts" / f"{key}.jpg", 90)
        out[key] = f"assets/attempts/{key}.jpg"
    return out


def build_dynamics() -> dict:
    """D(x) and D(G(z)) per epoch (how the duel goes) and FID at each evaluation."""
    runs = {"model1": "dcgan_portrait_64", "model2": "dcgan_diffaug_portrait_64",
            "model3": "sngan_diffaug_portrait_64", "final": "optuna26_portrait_64_s1"}
    out = {}
    for key, run in runs.items():
        h = pd.read_csv(RUNS / run / "history.csv")
        h = h[(h["epoch"] % 5 == 0) | (h["epoch"] == 1)]
        e = pd.read_csv(RUNS / run / "eval.csv")
        out[key] = {"epoch": h["epoch"].tolist(), "d_real": [r(v) for v in h["d_real"]],
                    "d_fake": [r(v) for v in h["d_fake"]], "loss_d": [r(v) for v in h["loss_d"]],
                    "eval_epoch": e["epoch"].tolist(), "fid": [r(v, 1) for v in e["FID"]],
                    "precision": [r(v) for v in e["precision"]], "recall": [r(v) for v in e["recall"]]}
    return out


def build_confirmation() -> dict:
    out = {}
    for setting in ("optuna26", "optuna27"):
        for seed in (42, 1, 2):
            e = pd.read_csv(RUNS / f"{setting}_portrait_64_s{seed}" / "eval.csv")
            out[f"{setting}_s{seed}"] = {"epoch": e["epoch"].tolist(), "fid": [r(v, 1) for v in e["FID"]]}
    return out


def build_optuna() -> dict:
    import optuna
    from optuna.importance import FanovaImportanceEvaluator

    t = pd.read_csv(ROOT / "reports" / "optuna" / "portrait_64_trials.csv")
    trials = [{"n": int(x["number"]), "value": r(x["value"], 1), "state": x["state"],
               "batch": int(x["params_batch_size"]), "n_dis": int(x["params_n_dis"]), "beta1": r(x["params_beta1"], 2),
               "sn": bool(x["params_spectral_norm"]), "loss": x["params_loss"], "diffaug": x["params_diffaug"]}
              for _, x in t.iterrows()]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name="portrait_64",
                              storage=f"sqlite:///{(ROOT / 'reports' / 'optuna' / 'portrait_64.db').as_posix()}")
    imps = pd.DataFrame([optuna.importance.get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=s))
                         for s in range(5)]).mean().sort_values(ascending=False)
    return {"trials": trials, "importances": [{"param": k, "value": r(v)} for k, v in imps.items()]}


# --------------------------------------------------------------------------- reference scale
def build_scale() -> dict:
    """Scores of the fake generators and of the models, portraits 64×80 (the inspector's scale)."""
    b = pd.read_csv(ROOT / "reports" / "metrics" / "baselines.csv")
    b = b[(b["genre"] == "portrait") & (b["résolution"] == "64×80")]
    keys = {"Copie du train": "copy", "Réel non vu (test)": "real_unseen", "Flou léger": "blur_light",
            "Bruit (σ=25)": "noise", "Mode collapse (20 images)": "collapse", "Flou fort": "blur_strong",
            "Bruit pur": "pure_noise", "Image moyenne": "mean"}
    cols = ["n", "FID", "FID_ntest", "KID_x1000", "IS", "precision", "recall", "density", "coverage", "mem_ratio", "copies"]
    fakes = [{"key": keys[x["baseline"]], "label": x["baseline"], **{c: r(x[c]) for c in cols}} for _, x in b.iterrows()]

    m = pd.read_csv(ROOT / "reports" / "metrics" / "models.csv")
    best = m[m["checkpoint"] == "best"].set_index("run")
    names = {"dcgan_portrait_64": "Modèle n°1 · DCGAN", "dcgan_diffaug_portrait_64": "Modèle n°2 · + DiffAugment",
             "sngan_diffaug_portrait_64": "Modèle n°3 · + norm. spectrale + hinge",
             "optuna26_portrait_64_s1": "Modèle final · réglage Optuna"}
    models = [{"key": k, "label": v, "epoch": int(best.loc[k, "epoch"]), **{c: r(best.loc[k, c]) for c in cols}}
              for k, v in names.items()]
    return {"fakes": fakes, "models": models}


def build_fake_examples() -> dict:
    """What each fake generator looks like: the same four training portraits, degraded."""
    rng = np.random.default_rng(0)
    imgs = ds.load_split("portrait", "train", (64, 80))[[5, 120, 900, 2100]]
    kinds = {
        "copy": imgs,
        "blur_light": np.stack([np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(1.0))) for x in imgs]),
        "noise": np.clip(imgs + rng.normal(0, 25, imgs.shape), 0, 255).astype(np.uint8),
        "collapse": np.stack([imgs[0]] * 4),
        "blur_strong": np.stack([np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(2.5))) for x in imgs]),
        "pure_noise": rng.integers(0, 256, imgs.shape, dtype=np.uint8),
    }
    out = {}
    for key, arr in kinds.items():
        save(strip(list(arr), 2, gap=4), ASSETS / "fakes" / f"{key}.jpg", 90)
        out[key] = f"assets/fakes/{key}.jpg"
    return out


# --------------------------------------------------------------------------- figures
def build_figures() -> dict:
    out = {}
    for name in FIGURE_FILES:
        im = Image.open(FIGURES / f"{name}.png").convert("RGB")
        if im.width > 1700:
            im = im.resize((1700, round(1700 * im.height / im.width)), Image.LANCZOS)
        save(im, ASSETS / "figs" / f"{name}.jpg", 88)
        out[name] = {"file": f"assets/figs/{name}.jpg", "w": im.width, "h": im.height}
    return out


def main() -> None:
    data = {"funnel": json.loads((ROOT / "data" / "datasets" / "summary.json").read_text(encoding="utf-8")),
            "cards": json.loads((ROOT / "models" / "models.json").read_text(encoding="utf-8"))}
    for key, fn in [("generated", build_generated), ("easel", build_easel), ("attempts", build_attempts),
                    ("dynamics", build_dynamics), ("confirmation", build_confirmation), ("optuna", build_optuna),
                    ("scale", build_scale), ("fakeExamples", build_fake_examples), ("figures", build_figures)]:
        data[key] = fn()
        print(f"{key}: done", flush=True)
    js = HERE / "js" / "data.js"
    js.parent.mkdir(exist_ok=True)
    js.write_text("// Generated by build_assets.py — do not edit by hand.\nwindow.GANISME = "
                  + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    size = sum(f.stat().st_size for f in ASSETS.rglob("*") if f.is_file()) / 1e6
    print(f"written: {js.relative_to(ROOT)} | assets: {size:.1f} MB")


if __name__ == "__main__":
    main()
