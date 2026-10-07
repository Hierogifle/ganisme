"""Confirm the best settings found by Optuna.

The Optuna study (notebook 07) has three limits: a single training run per trial, only 300 epochs,
and no checkpoint saved. This script retrains the best trials:

  - for longer (600 epochs by default): the best trials were still improving at epoch 300;
  - with several seeds: to tell a real gain from a lucky draw;
  - with the full evaluation and checkpoints: to get a usable model.

Hyperparameters are read directly from the study database (nothing is copied by hand).
Runs are interleaved (trial A seed 1, trial B seed 1, trial A seed 2, …): if the computation is
interrupted, a comparison between the settings is already available. A finished run is never
restarted: running the same command again resumes where the computation stopped.

Example:
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
    """A run is finished if its last evaluation is at the last epoch and it has its checkpoints."""
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

    plan = [(n, s) for s in args.seeds for n in args.trials]          # interleaved by seed
    print(f"{len(plan)} training runs of {args.epochs} epochs: {plan}", flush=True)
    for i, (n, seed) in enumerate(plan, 1):
        name = f"optuna{n}_{args.genre}_{args.res}_s{seed}"
        if is_done(name, args.epochs):
            print(f"[{i}/{len(plan)}] {name}: already finished, skipped", flush=True)
            continue
        print(f"[{i}/{len(plan)}] {name}: {params[n]}", flush=True)
        t0 = time.time()
        tr.train(tr.Config(name=name, genre=args.genre, res=args.res, epochs=args.epochs, seed=seed,
                           eval_every=25, **params[n]))
        print(f"[{i}/{len(plan)}] {name} finished in {(time.time() - t0) / 60:.0f} min", flush=True)
    print("Confirmation finished.", flush=True)


if __name__ == "__main__":
    main()
