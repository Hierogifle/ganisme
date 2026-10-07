"""Exporte les modèles retenus pour l'application Streamlit.

Les checkpoints d'entraînement (runs/<run>/checkpoints/best.pt) contiennent le générateur ET le discriminateur.
Pour générer des images, seul le générateur sert : on l'exporte seul, ce qui divise la taille des fichiers
par deux (le plus gros passe de 98 Mo à 53 Mo, sous la limite de 100 Mo par fichier de GitHub).

Sorties :
    models/<clé>.pt       poids du générateur + configuration + scores du checkpoint
    models/models.json    fiche descriptive de chaque modèle, lue par l'application

    python src/export_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS, OUT = ROOT / "runs", ROOT / "models"

# clé -> (run retenu, notebook qui justifie le choix)
SELECTED = {
    "portrait_64": ("optuna26_portrait_64_s1", "08"),
    "portrait_128": ("portrait_128_w64_s42", "09"),
    "landscape_64": ("landscape_64_best_s42", "10"),
    "landscape_128": ("landscape_128_w64_s42", "10"),
}
GENRE_LABEL = {"portrait": "Portraits", "landscape": "Paysages"}
KEEP_SCORES = ["FID", "FID_ntest", "KID_x1000", "precision", "recall", "density", "coverage", "mem_ratio", "copies"]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cards = {}
    for key, (run, notebook) in SELECTED.items():
        ckpt = torch.load(RUNS / run / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
        cfg, scores = ckpt["config"], ckpt["scores"]
        torch.save({"G": ckpt["G"], "config": cfg, "epoch": ckpt["epoch"], "scores": scores}, OUT / f"{key}.pt")
        hist = pd.read_csv(RUNS / run / "history.csv")
        meta = pd.read_csv(ROOT / "data" / "datasets" / cfg["genre"] / "metadata.csv")
        cards[key] = {
            "genre": cfg["genre"], "genre_label": GENRE_LABEL[cfg["genre"]],
            "width": cfg["size"][0], "height": cfg["size"][1],
            "run": run, "notebook": notebook, "epoch": int(ckpt["epoch"]),
            "n_train": int((meta["SPLIT"] == "train").sum()),
            "params_G": int(sum(v.numel() for k, v in ckpt["G"].items() if "num_batches_tracked" not in k and "running" not in k)),
            "train_minutes": round(float(hist["seconds"].sum()) / 60),      # temps de calcul, hors évaluations
            "hyperparams": {k: cfg[k] for k in ["loss", "spectral_norm", "lr_g", "lr_d", "beta1", "beta2", "n_dis",
                                                "diffaug", "batch_size", "nz", "ngf", "n_up", "epochs", "seed"]},
            "scores": {k: round(float(scores[k]), 4) for k in KEEP_SCORES},
        }
        size = (OUT / f"{key}.pt").stat().st_size / 1e6
        print(f"{key:14s} <- {run:28s} époque {ckpt['epoch']:3d} | FID {scores['FID']:.1f} | {size:.0f} Mo")
    (OUT / "models.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    print("écrit : models/models.json")


if __name__ == "__main__":
    main()
