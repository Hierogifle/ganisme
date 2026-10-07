"""Confirmation des meilleurs réglages trouvés par Optuna.

L'étude Optuna (notebook 07) a trois limites : un seul entraînement par essai, 300 époques seulement,
et aucun checkpoint sauvegardé. Ce script ré-entraîne les meilleurs essais :

  - plus longtemps (600 époques par défaut) : les meilleurs essais progressaient encore à l'époque 300 ;
  - avec plusieurs graines : pour distinguer un vrai gain d'un coup de chance ;
  - avec l'évaluation complète et les checkpoints : pour obtenir un modèle utilisable.

Les hyperparamètres sont relus directement dans la base de l'étude (aucune recopie à la main).
Les runs sont entrelacés (essai A graine 1, essai B graine 1, essai A graine 2, …) : si le calcul est
interrompu, on dispose déjà d'une comparaison entre les réglages. Un run terminé n'est jamais relancé :
relancer la même commande reprend là où le calcul s'est arrêté.

Exemple :
    python src/confirm_best.py --trials 26 27 --seeds 42 1 2 --epochs 600
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import optuna
import pandas as pd

import train as tr

ROOT = Path(__file__).resolve().parents[1]


def is_done(name: str, epochs: int) -> bool:
    """Un run est terminé si sa dernière évaluation porte sur la dernière époque et qu'il a ses checkpoints."""
    run = tr.RUNS / name
    if not (run / "eval.csv").exists() or not (run / "checkpoints" / "best.pt").exists():
        return False
    return int(pd.read_csv(run / "eval.csv")["epoch"].iloc[-1]) >= epochs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", default="portrait_64")
    p.add_argument("--genre", default="portrait")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--trials", type=int, nargs="+", default=[26, 27])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    p.add_argument("--epochs", type=int, default=600)
    args = p.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=args.study,
                              storage=f"sqlite:///{(ROOT / 'reports' / 'optuna' / f'{args.study}.db').as_posix()}")
    params = {t.number: t.params for t in study.trials if t.number in args.trials}

    plan = [(n, s) for s in args.seeds for n in args.trials]          # entrelacé par graine
    print(f"{len(plan)} entraînements de {args.epochs} époques : {plan}", flush=True)
    for i, (n, seed) in enumerate(plan, 1):
        name = f"optuna{n}_{args.genre}_{args.res}_s{seed}"
        if is_done(name, args.epochs):
            print(f"[{i}/{len(plan)}] {name} : déjà terminé, ignoré", flush=True)
            continue
        print(f"[{i}/{len(plan)}] {name} : {params[n]}", flush=True)
        t0 = time.time()
        tr.train(tr.Config(name=name, genre=args.genre, res=args.res, epochs=args.epochs, seed=seed,
                           eval_every=25, **params[n]))
        print(f"[{i}/{len(plan)}] {name} terminé en {(time.time() - t0) / 60:.0f} min", flush=True)
    print("Confirmation terminée.", flush=True)


if __name__ == "__main__":
    main()
