"""Landscapes: second genre, same approach as for portraits, in three steps.

The whole journey (3 models + Optuna) is not repeated: the point is to check whether what was learned
on portraits transfers to another genre, and then to move up in resolution.

  Stage 1 — 64×48
    a. `landscape_64_dcgan_s42`  baseline DCGAN, hyperparameters of the paper (equivalent of model 1):
                                 the starting point, to measure the gain.
    b. `landscape_64_best_s42`   hyperparameters found by Optuna on portraits (setting 26), reused
                                 as they are: do they transfer to another genre?
  Stage 2 — 128×96
    c. `landscape_128_w64_s42`   same setting, one more layer, width 64 (the architecture selected for
                                 portraits at 128×160).

4:3 horizontal format: the starting grid of the generator is 3×4 (height × width) instead of 5×4.

Resumable: after an interruption, running the same command again resumes where the computation stopped.

    python src/run_landscape.py --stage 1
    python src/run_landscape.py --stage 2
"""
from __future__ import annotations

import argparse
import time

import train as tr
from run_portrait_128 import BEST_64

QUEUE = {
    1: [dict(name="landscape_64_dcgan_s42", res=64, n_up=4, epochs=500),              # default settings = DCGAN
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
        print(f"=== [{i}/{len(jobs)}] {cfg.name}: done ({(time.time() - t0) / 60:.0f} min in this session)", flush=True)
    print("=== Queue finished.", flush=True)


if __name__ == "__main__":
    main()
