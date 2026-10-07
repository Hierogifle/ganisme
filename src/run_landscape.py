"""Paysages : second genre, même démarche que pour les portraits, en trois étapes.

On ne refait pas tout le parcours (3 modèles + Optuna) : on vérifie si ce qui a été appris sur les portraits
se transpose à un autre genre, puis on monte en résolution.

  Étape 1 — 64×48
    a. `landscape_64_dcgan_s42`  DCGAN de base, hyperparamètres de l'article (équivalent du modèle n°1) :
                                 le point de départ, pour mesurer le gain.
    b. `landscape_64_best_s42`   hyperparamètres trouvés par Optuna sur les portraits (réglage n°26),
                                 repris tels quels : se transposent-ils à un autre genre ?
  Étape 2 — 128×96
    c. `landscape_128_w64_s42`   même réglage, une couche de plus, largeur 64 (architecture retenue
                                 pour les portraits en 128×160).

Format 4:3 horizontal : la grille de départ du générateur est 3×4 (hauteur × largeur) au lieu de 5×4.

Reprenable : relancer la même commande après une interruption reprend là où le calcul s'est arrêté.

    python src/run_landscape.py --stage 1
    python src/run_landscape.py --stage 2
"""
from __future__ import annotations

import argparse
import time

import train as tr
from run_portrait_128 import BEST_64

QUEUE = {
    1: [dict(name="landscape_64_dcgan_s42", res=64, n_up=4, epochs=500),              # réglages par défaut = DCGAN
        dict(name="landscape_64_best_s42", res=64, n_up=4, epochs=600, **BEST_64)],
    2: [dict(name="landscape_128_w64_s42", res=128, n_up=5, epochs=600, **BEST_64)],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    args = p.parse_args()

    jobs = [j for s in args.stage for j in QUEUE[s]]
    for i, job in enumerate(jobs, 1):
        cfg = tr.Config(genre="landscape", seed=42, eval_every=25, resume=True, **job)
        print(f"=== [{i}/{len(jobs)}] {cfg.name}", flush=True)
        t0 = time.time()
        tr.train(cfg)
        print(f"=== [{i}/{len(jobs)}] {cfg.name} : fin ({(time.time() - t0) / 60:.0f} min dans cette session)", flush=True)
    print("=== File terminée.", flush=True)


if __name__ == "__main__":
    main()
