"""Hyperparameter search with Optuna.

Each trial trains a full GAN with a combination of hyperparameters proposed by Optuna, then returns
a score to minimise. Optuna uses the **TPE** algorithm (Tree-structured Parzen Estimator): from past
trials, it models which regions of the search space give good and bad scores, and proposes the next
trials where the "good / bad" ratio is highest. This is far more efficient than a grid or a random
search when each trial is expensive.

Choices specific to GANs (discussed in notebook 06 and in the notebook analysing the study):

1. **Objective robust to collapses**: mean FID of the last 3 evaluations, not the best FID. A setting
   that reaches a good score and then collapses (like model 2) is penalised; a stable setting is
   favoured.
2. **Cautious pruning**: Optuna stops a trial only if it is in the **worst quarter** at the same
   epoch, and never before epoch 125 — so that settings which converge slowly (such as spectral
   normalisation) are not wrongly eliminated.
3. **Same seed for every trial**: differences come from the hyperparameters, not from the
   initialisation.
4. **Light evaluation** (FID, precision/recall, memorisation) every 25 epochs.
5. **First trials imposed**: the settings of models 2 and 3, plus two settings known from the
   literature, serve as landmarks for the algorithm.

The study is stored in an SQLite database: if the computer stops, running the same command again
resumes the study where it left off.

Example:
    python src/optuna_search.py --n-trials 30 --timeout-hours 8
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

import train as tr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "optuna"

DIFFAUG_POLICIES = ["", "translation,cutout", "color,translation", "color,translation,cutout"]

# Reference settings evaluated first (landmarks for TPE)
ENQUEUED = [
    # Model 2: DCGAN + DiffAugment, hyperparameters of the DCGAN paper
    {"loss": "bce", "spectral_norm": False, "lr_g": 2e-4, "lr_d": 2e-4, "beta1": 0.5, "beta2": 0.999,
     "n_dis": 1, "diffaug": "color,translation,cutout", "batch_size": 128},
    # Model 3: SNGAN + DiffAugment with the same hyperparameters (discriminator too weak)
    {"loss": "hinge", "spectral_norm": True, "lr_g": 2e-4, "lr_d": 2e-4, "beta1": 0.5, "beta2": 0.999,
     "n_dis": 1, "diffaug": "color,translation,cutout", "batch_size": 128},
    # SNGAN + TTUR (Heusel et al., 2017): D learns 4× faster than G, Adam(0, 0.9)
    {"loss": "hinge", "spectral_norm": True, "lr_g": 1e-4, "lr_d": 4e-4, "beta1": 0.0, "beta2": 0.9,
     "n_dis": 1, "diffaug": "color,translation,cutout", "batch_size": 128},
    # SNGAN "à la Miyato": several D updates per G update, Adam(0, 0.9)
    {"loss": "hinge", "spectral_norm": True, "lr_g": 2e-4, "lr_d": 2e-4, "beta1": 0.0, "beta2": 0.9,
     "n_dis": 3, "diffaug": "color,translation,cutout", "batch_size": 64},
]


def suggest(trial: optuna.Trial) -> dict:
    """The search space: the levers that set the balance between G and D."""
    return {
        # Loss and spectral normalisation: chosen independently (4 combinations)
        "loss": trial.suggest_categorical("loss", ["bce", "hinge"]),
        "spectral_norm": trial.suggest_categorical("spectral_norm", [False, True]),
        # Separate learning rates (log scale): their ratio sets the relative "speed" of D and G
        "lr_g": trial.suggest_float("lr_g", 5e-5, 5e-4, log=True),
        "lr_d": trial.suggest_float("lr_d", 5e-5, 1e-3, log=True),
        # Adam: beta1 = 0.5 (DCGAN) or 0 (SNGAN, BigGAN); beta2 = 0.999 (default) or 0.9 (SNGAN)
        "beta1": trial.suggest_categorical("beta1", [0.0, 0.5]),
        "beta2": trial.suggest_categorical("beta2", [0.9, 0.999]),
        # D updates per G update (5 in the SNGAN paper, too expensive here)
        "n_dis": trial.suggest_categorical("n_dis", [1, 2, 3]),
        "diffaug": trial.suggest_categorical("diffaug", DIFFAUG_POLICIES),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
    }


def make_objective(args):
    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        cfg = tr.Config(name=f"optuna/{args.study}/trial_{trial.number:03d}", genre=args.genre, res=args.res,
                        epochs=args.epochs, eval_every=25, eval_light=True, save_checkpoints=False,
                        register=False, seed=42, **params)
        fids = []

        def on_eval(epoch: int, scores: dict) -> None:
            fids.append(scores["FID"])
            trial.set_user_attr("last_precision", scores["precision"])
            trial.set_user_attr("last_recall", scores["recall"])
            trial.set_user_attr("last_mem_ratio", scores["mem_ratio"])
            # At each evaluation, Optuna decides whether this trial should be stopped
            trial.report(scores["FID"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"pruned at epoch {epoch} (FID {scores['FID']:.1f})")

        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):      # per-trial log kept in runs/optuna/…
                tr.train(cfg, on_eval=on_eval)
        except FloatingPointError as exc:                      # numerical divergence: very bad score
            trial.set_user_attr("error", str(exc))
            return 999.0
        finally:
            trial.set_user_attr("minutes", round((time.time() - t0) / 60, 1))
            trial.set_user_attr("best_fid", float(min(fids)) if fids else None)
            trial.set_user_attr("n_evals", len(fids))
            torch.cuda.empty_cache()

        # Objective: mean of the last 3 evaluations (penalises collapses at the end of training)
        tail = fids[-3:]
        trial.set_user_attr("fid_trend", float(fids[-1] - fids[-3]) if len(fids) >= 3 else None)
        return float(np.mean(tail))

    return objective


def export(study: optuna.Study) -> None:
    """Export the trials to CSV and the best setting to JSON (read by the analysis notebook)."""
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    df.to_csv(OUT / f"{study.study_name}_trials.csv", index=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if done:
        best = study.best_trial
        (OUT / f"{study.study_name}_best.json").write_text(
            json.dumps({"number": best.number, "value": best.value, "params": best.params,
                        "user_attrs": best.user_attrs}, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", default="portrait_64")
    p.add_argument("--genre", default="portrait")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--timeout-hours", type=float, default=8.0)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study, storage=f"sqlite:///{(OUT / f'{args.study}.db').as_posix()}", load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=8),
        # Prune if the trial is in the worst quarter (75th percentile) at the same epoch, after epoch 125
        pruner=optuna.pruners.PercentilePruner(75.0, n_startup_trials=5, n_warmup_steps=125),
    )
    if len(study.trials) == 0:
        for params in ENQUEUED:
            study.enqueue_trial(params)

    def log(study, trial):
        export(study)
        v = f"{trial.value:.1f}" if trial.value is not None else "-"
        best = f"{study.best_value:.1f} (trial {study.best_trial.number})" if any(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else "-"
        print(f"trial {trial.number:3d} | {trial.state.name:8s} | objective {v:>6s} | "
              f"{trial.user_attrs.get('minutes', '?')} min | best {best} | {trial.params}", flush=True)

    already = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.n_trials - already)
    print(f"Study '{args.study}': {already} trials finished, {remaining} to run "
          f"({args.epochs} epochs per trial, limit {args.timeout_hours} h)", flush=True)
    study.optimize(make_objective(args), n_trials=remaining, timeout=args.timeout_hours * 3600, callbacks=[log])
    export(study)
    print("Done. Best trial:", study.best_trial.number, study.best_value, study.best_params, flush=True)


if __name__ == "__main__":
    main()
