"""Recherche d'hyperparamètres avec Optuna.

Chaque essai (« trial ») entraîne un GAN complet avec une combinaison d'hyperparamètres proposée par
Optuna, puis renvoie un score à minimiser. Optuna utilise l'algorithme **TPE** (Tree-structured Parzen
Estimator) : il modélise, à partir des essais passés, les zones de l'espace de recherche qui donnent de
bons et de mauvais scores, et propose les essais suivants là où le ratio « bon / mauvais » est le plus
élevé. C'est bien plus efficace qu'une grille ou qu'un tirage aléatoire quand chaque essai coûte cher.

Choix spécifiques aux GAN (discutés dans le notebook 06 et le notebook d'analyse de l'étude) :

1. **Objectif robuste aux effondrements** : moyenne du FID des 3 dernières évaluations, et non le meilleur
   FID. Un réglage qui atteint un bon score puis s'effondre (comme le modèle n°2) est pénalisé ; un
   réglage stable est favorisé.
2. **Élagage prudent** : Optuna arrête un essai seulement s'il fait partie du **quart le moins bon** à la
   même époque, et jamais avant l'époque 125 — pour ne pas éliminer à tort des réglages qui convergent
   lentement (comme la normalisation spectrale).
3. **Même graine pour tous les essais** : les écarts viennent des hyperparamètres, pas de l'initialisation.
4. **Évaluation allégée** (FID, précision/rappel, mémorisation) toutes les 25 époques.
5. **Premiers essais imposés** : les réglages des modèles n°2 et n°3, ainsi que deux réglages connus de la
   littérature, servent de points de repère à l'algorithme.

L'étude est stockée dans une base SQLite : si l'ordinateur s'arrête, relancer la même commande reprend
l'étude là où elle en était.

Exemple :
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

# Réglages de référence évalués en premier (points de repère pour TPE)
ENQUEUED = [
    # Modèle n°2 : DCGAN + DiffAugment, hyperparamètres de l'article DCGAN
    {"loss": "bce", "spectral_norm": False, "lr_g": 2e-4, "lr_d": 2e-4, "beta1": 0.5, "beta2": 0.999,
     "n_dis": 1, "diffaug": "color,translation,cutout", "batch_size": 128},
    # Modèle n°3 : SNGAN + DiffAugment avec les mêmes hyperparamètres (discriminateur trop faible)
    {"loss": "hinge", "spectral_norm": True, "lr_g": 2e-4, "lr_d": 2e-4, "beta1": 0.5, "beta2": 0.999,
     "n_dis": 1, "diffaug": "color,translation,cutout", "batch_size": 128},
    # SNGAN + TTUR (Heusel et al., 2017) : D apprend 4× plus vite que G, Adam(0 ; 0,9)
    {"loss": "hinge", "spectral_norm": True, "lr_g": 1e-4, "lr_d": 4e-4, "beta1": 0.0, "beta2": 0.9,
     "n_dis": 1, "diffaug": "color,translation,cutout", "batch_size": 128},
    # SNGAN « à la Miyato » : plusieurs mises à jour de D par mise à jour de G, Adam(0 ; 0,9)
    {"loss": "hinge", "spectral_norm": True, "lr_g": 2e-4, "lr_d": 2e-4, "beta1": 0.0, "beta2": 0.9,
     "n_dis": 3, "diffaug": "color,translation,cutout", "batch_size": 64},
]


def suggest(trial: optuna.Trial) -> dict:
    """L'espace de recherche : les leviers qui règlent l'équilibre entre G et D."""
    return {
        # Perte et normalisation spectrale : choisies indépendamment (4 combinaisons)
        "loss": trial.suggest_categorical("loss", ["bce", "hinge"]),
        "spectral_norm": trial.suggest_categorical("spectral_norm", [False, True]),
        # Taux d'apprentissage séparés (échelle log) : leur rapport règle la « vitesse » relative de D et G
        "lr_g": trial.suggest_float("lr_g", 5e-5, 5e-4, log=True),
        "lr_d": trial.suggest_float("lr_d", 5e-5, 1e-3, log=True),
        # Adam : β1 = 0,5 (DCGAN) ou 0 (SNGAN, BigGAN) ; β2 = 0,999 (défaut) ou 0,9 (SNGAN)
        "beta1": trial.suggest_categorical("beta1", [0.0, 0.5]),
        "beta2": trial.suggest_categorical("beta2", [0.9, 0.999]),
        # Mises à jour de D par mise à jour de G (5 dans l'article SNGAN, trop coûteux ici)
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
            # Optuna décide, à chaque évaluation, s'il faut arrêter cet essai
            trial.report(scores["FID"], step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"élagué à l'époque {epoch} (FID {scores['FID']:.1f})")

        t0 = time.time()
        try:
            with contextlib.redirect_stdout(io.StringIO()):      # journal par essai dans runs/optuna/…
                tr.train(cfg, on_eval=on_eval)
        except FloatingPointError as exc:                      # divergence numérique : très mauvais score
            trial.set_user_attr("error", str(exc))
            return 999.0
        finally:
            trial.set_user_attr("minutes", round((time.time() - t0) / 60, 1))
            trial.set_user_attr("best_fid", float(min(fids)) if fids else None)
            trial.set_user_attr("n_evals", len(fids))
            torch.cuda.empty_cache()

        # Objectif : moyenne des 3 dernières évaluations (pénalise les effondrements de fin d'entraînement)
        tail = fids[-3:]
        trial.set_user_attr("fid_trend", float(fids[-1] - fids[-3]) if len(fids) >= 3 else None)
        return float(np.mean(tail))

    return objective


def export(study: optuna.Study) -> None:
    """Exporte les essais en CSV et le meilleur réglage en JSON (lus par le notebook d'analyse)."""
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
        # Élague si l'essai est dans le quart le moins bon (75e centile) à la même époque, après l'époque 125
        pruner=optuna.pruners.PercentilePruner(75.0, n_startup_trials=5, n_warmup_steps=125),
    )
    if len(study.trials) == 0:
        for params in ENQUEUED:
            study.enqueue_trial(params)

    def log(study, trial):
        export(study)
        v = f"{trial.value:.1f}" if trial.value is not None else "-"
        best = f"{study.best_value:.1f} (essai {study.best_trial.number})" if any(
            t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) else "-"
        print(f"essai {trial.number:3d} | {trial.state.name:8s} | objectif {v:>6s} | "
              f"{trial.user_attrs.get('minutes', '?')} min | meilleur {best} | {trial.params}", flush=True)

    already = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.n_trials - already)
    print(f"Étude « {args.study} » : {already} essais terminés, {remaining} à lancer "
          f"({args.epochs} époques par essai, limite {args.timeout_hours} h)", flush=True)
    study.optimize(make_objective(args), n_trials=remaining, timeout=args.timeout_hours * 3600, callbacks=[log])
    export(study)
    print("Terminé. Meilleur essai :", study.best_trial.number, study.best_value, study.best_params, flush=True)


if __name__ == "__main__":
    main()
