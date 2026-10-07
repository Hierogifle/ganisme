"""Portraits at 128×160: resumable queue of training runs.

Two architectures are compared, with the hyperparameters of the final 64×80 model (Optuna setting 26,
notebook 08). Going from 64×80 to 128×160 adds one doubling layer to the generator and one reduction
layer to the discriminator (n_up = 5, still starting from a 5×4 grid). There are two ways to do it:

  - width 64: 64 filters are kept at the highest resolution; the added layer, on the 5×4 grid side,
    has 1,024 channels. The network is larger (13.2 M parameters for G);
  - width 32: all widths are halved. The network has the same size as the 64×80 model (3.8 M
    parameters) and trains almost as fast.

The script is **resumable**: each training run saves its full state at every evaluation, and a
finished run is never restarted. After an interruption (sleep, shutdown, closed session), running
the same command again resumes where the computation stopped:

    python src/run_portrait_128.py
"""
from __future__ import annotations

import argparse
import time

import train as tr

# Hyperparameters of the final 64×80 model (Optuna trial 26)
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
        print(f"=== [{i}/{len(QUEUE)}] {cfg.name}: done ({(time.time() - t0) / 60:.0f} min in this session)", flush=True)
    print("=== Queue finished.", flush=True)


if __name__ == "__main__":
    main()
