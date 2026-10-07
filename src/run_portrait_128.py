"""Portraits en 128×160 : file d'entraînements reprenable.

Deux architectures sont comparées, avec les hyperparamètres du modèle final 64×80 (réglage Optuna n°26,
notebook 08). Passer de 64×80 à 128×160 ajoute une couche de doublement au générateur et une couche de
réduction au discriminateur (n_up = 5, toujours à partir d'une grille 5×4). Deux façons de le faire :

  - largeur 64 : on garde 64 filtres à la résolution la plus haute ; la couche ajoutée, côté grille 5×4,
    a 1 024 canaux. Le réseau est plus grand (13,2 M de paramètres pour G) ;
  - largeur 32 : on divise toutes les largeurs par deux. Le réseau a la même taille que le modèle 64×80
    (3,8 M de paramètres) et s'entraîne presque aussi vite.

Le script est **reprenable** : chaque entraînement sauvegarde son état complet à chaque évaluation, et un
entraînement terminé n'est jamais relancé. Après une interruption (veille, arrêt du PC, fermeture de la
session), relancer la même commande reprend là où le calcul s'est arrêté :

    python src/run_portrait_128.py
"""
from __future__ import annotations

import argparse
import time

import train as tr

# Hyperparamètres du modèle final 64×80 (essai Optuna n°26)
BEST_64 = dict(loss="bce", spectral_norm=True, lr_g=0.0003041101671322451, lr_d=0.0004984335369118899,
               beta1=0.0, beta2=0.999, n_dis=2, diffaug="color,translation", batch_size=64)

QUEUE = [
    dict(name="portrait_128_w64_s42", ngf=64, ndf=64, seed=42),
    dict(name="portrait_128_w32_s42", ngf=32, ndf=32, seed=42),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=600)
    args = p.parse_args()

    for i, job in enumerate(QUEUE, 1):
        cfg = tr.Config(genre="portrait", res=128, n_up=5, epochs=args.epochs, eval_every=25, resume=True,
                        **BEST_64, **job)
        print(f"=== [{i}/{len(QUEUE)}] {cfg.name}", flush=True)
        t0 = time.time()
        tr.train(cfg)
        print(f"=== [{i}/{len(QUEUE)}] {cfg.name} : fin ({(time.time() - t0) / 60:.0f} min dans cette session)", flush=True)
    print("=== File terminée.", flush=True)


if __name__ == "__main__":
    main()
