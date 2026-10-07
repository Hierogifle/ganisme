"""Train a GAN on one genre, with metric tracking.

Each iteration alternates two updates (see the literature review, question 6):

  1. Discriminator D — it is shown a batch of real images (target 1) and a batch of generated images
     (target 0); binary cross-entropy is minimised. Generated images are produced without gradient:
     this step does not modify the generator.
  2. Generator G — a new batch is generated and passed through D; the target is 1 ("make D believe
     it is real"). This is the **non-saturating** loss -log D(G(z)): it gives strong gradients even
     when D easily rejects the fakes, unlike log(1 - D(G(z))).

Every `eval_every` epochs:
  - a grid of images generated from **fixed** noise (the same "painting" can be watched evolving);
  - every metric of `metrics.evaluate` on N_EVAL images (N_EVAL = size of the training set);
  - a checkpoint; the best one (lowest FID) is kept separately.

Outputs: runs/<name>/
    config.json      hyperparameters
    history.csv      losses and D outputs, averaged per epoch
    eval.csv         metrics at each evaluation
    samples/         image grids from fixed noise
    checkpoints/     best.pt (best FID) and last.pt
Plus one row per model in reports/metrics/models.csv (comparison table of the project).

Example:
    python src/train.py --genre portrait --res 64 --epochs 500 --name dcgan_portrait_64
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import dataset as ds
import metrics as mt
from diffaugment import diff_augment
from models import Discriminator, Generator, base_grid, count_params, dcgan_init

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REGISTRY = ROOT / "reports" / "metrics" / "models.csv"


@dataclass
class Config:
    name: str = "dcgan_portrait_64"
    genre: str = "portrait"
    res: int = 64                  # width; the height follows from the aspect ratio of the genre
    epochs: int = 500
    batch_size: int = 128
    lr_g: float = 2e-4             # values of the DCGAN paper
    lr_d: float = 2e-4
    beta1: float = 0.5             # beta1 = 0.5 instead of 0.9: less momentum, more stable for GANs
    beta2: float = 0.999
    nz: int = 100
    ngf: int = 64
    ndf: int = 64
    n_up: int = 4
    flip: bool = True              # augmentation: random horizontal flip
    real_label: float = 1.0        # 0.9 = label smoothing (optional, off for the baseline DCGAN)
    diffaug: str = ""              # DiffAugment policy, e.g. "color,translation,cutout" ("" = off)
    loss: str = "bce"              # "bce" (DCGAN, non-saturating) or "hinge" (SNGAN)
    spectral_norm: bool = False    # spectral normalisation in D (replaces its BatchNorm)
    n_dis: int = 1                 # D updates per G update (5 in the SNGAN paper)
    eval_every: int = 25
    eval_light: bool = False       # lighter evaluation (hyperparameter search), see metrics.evaluate
    save_checkpoints: bool = True
    register: bool = True          # add the run to the comparison table reports/metrics/models.csv
    resume: bool = False           # resume an interrupted training run (see checkpoints/resume.pt)
    n_fixed: int = 64              # number of images in the tracking grid
    seed: int = 42
    size: tuple[int, int] = field(default=(0, 0))   # (width, height), computed

    def resolve(self) -> "Config":
        cfg = ds.GENRES[self.genre]
        self.size = (self.res, round(self.res / cfg.aspect))
        return self


# --------------------------------------------------------------------------- data
def load_train_tensor(cfg: Config, device: str) -> torch.Tensor:
    """All training images in GPU memory (uint8, N×3×H×W): 3,448 images of 64×80 weigh only 53 MB,
    so a DataLoader re-reading the files at every epoch would be pointless."""
    x = ds.load_split(cfg.genre, "train", cfg.size)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)


def to_model_range(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1], the output range of the generator's Tanh."""
    return x_uint8.float().div_(127.5).sub_(1.0)


def batches(data: torch.Tensor, batch_size: int, flip: bool, gen: torch.Generator):
    """Reshuffled at every epoch; the last incomplete batch is dropped (BatchNorm dislikes small batches)."""
    idx = torch.randperm(len(data), generator=gen, device="cpu").to(data.device)
    for i in range(0, len(idx) - batch_size + 1, batch_size):
        x = to_model_range(data[idx[i:i + batch_size]])
        if flip:
            mask = torch.rand(len(x), device=x.device) < 0.5
            x[mask] = x[mask].flip(3)       # left-right flip for half of the images
        yield x


# --------------------------------------------------------------------------- generation / evaluation
@torch.no_grad()
def generate(G: Generator, z: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """uint8 images from latent vectors. G is switched to eval mode: BatchNorm uses its running
    statistics, as it will in the application (where images are generated one at a time)."""
    was_training = G.training
    G.eval()
    out = torch.cat([mt.to_uint8(G(z[i:i + batch_size])) for i in range(0, len(z), batch_size)])
    G.train(was_training)
    return out


def evaluation_latents(n: int, nz: int, device: str, seed: int = 42) -> torch.Tensor:
    """The SAME latent vectors for every evaluation and every model (protocol of notebook 03)."""
    return torch.randn(n, nz, generator=torch.Generator().manual_seed(seed)).to(device)


# --------------------------------------------------------------------------- losses
def d_loss(logit_real: torch.Tensor, logit_fake: torch.Tensor, kind: str, real_label: float = 1.0) -> torch.Tensor:
    """Discriminator loss.

    - bce   : binary cross-entropy, targets 1 (real) and 0 (fake) — the loss of the original GAN.
    - hinge : max(0, 1 − D(x)) + max(0, 1 + D(G(z))). D must push real scores above +1 and fake
              scores below −1, and then **stops being penalised**: it gains nothing from becoming
              "infinitely confident", which limits its dominance (together with spectral normalisation).
    """
    if kind == "hinge":
        return F.relu(1 - logit_real).mean() + F.relu(1 + logit_fake).mean()
    return F.binary_cross_entropy_with_logits(logit_real, torch.full_like(logit_real, real_label)) + \
        F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))


def g_loss(logit_gen: torch.Tensor, kind: str) -> torch.Tensor:
    """Generator loss: non-saturating bce −log D(G(z)), or hinge −D(G(z)) (raise the score of the fakes)."""
    if kind == "hinge":
        return -logit_gen.mean()
    return F.binary_cross_entropy_with_logits(logit_gen, torch.ones_like(logit_gen))


# --------------------------------------------------------------------------- resume
# Hyperparameters that must be identical for a resume to make sense
RESUME_KEYS = ["genre", "res", "nz", "ngf", "ndf", "n_up", "loss", "spectral_norm", "lr_g", "lr_d", "beta1", "beta2",
               "n_dis", "diffaug", "batch_size", "flip", "real_label", "seed"]


def is_complete(cfg: Config) -> bool:
    """True if the run has already reached its last epoch and has its checkpoints."""
    run = RUNS / cfg.name
    if not (run / "eval.csv").exists() or not (run / "checkpoints" / "best.pt").exists():
        return False
    return int(pd.read_csv(run / "eval.csv")["epoch"].iloc[-1]) >= cfg.epochs


def save_resume_state(path: Path, **state) -> None:
    """Write the full training state. Atomic write: the state goes to a temporary file which is then
    renamed, so a half-written file is never left behind if the computer stops at that moment."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------- training
def train(cfg: Config, on_eval=None) -> Path:
    """Train a GAN according to `cfg`.

    `on_eval(epoch, scores)` is called after each evaluation. It may raise an exception to stop the
    training run — this is how Optuna prunes unpromising trials.

    Resume (`cfg.resume`): at each evaluation, the full state is saved to `checkpoints/resume.pt` —
    weights of both networks, state of both Adam optimisers (their moving averages), random number
    generators, evaluation history. If training is interrupted (sleep, shutdown, closed session),
    running the same command again restarts from the last evaluation: at most `eval_every` epochs
    are lost. A run that is already complete is not restarted.
    A resumed run is not bit-for-bit identical to an uninterrupted one (the GPU is not deterministic),
    but it is statistically equivalent.
    """
    cfg.resolve()
    run = RUNS / cfg.name
    if cfg.resume and is_complete(cfg):
        print(f"[{cfg.name}] already complete ({cfg.epochs} epochs), skipped", flush=True)
        return run
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    (run / "samples").mkdir(parents=True, exist_ok=True)
    if cfg.save_checkpoints:
        (run / "checkpoints").mkdir(exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    # Data and reference features for the metrics
    data = load_train_tensor(cfg, device)
    ref = mt.Reference.load_or_compute(cfg.genre, cfg.size, ds.DATASETS / cfg.genre)
    n_eval = len(ref.train)

    # Models
    base = base_grid(cfg.size, cfg.n_up)
    G = Generator(cfg.nz, cfg.ngf, base, cfg.n_up).to(device)
    D = Discriminator(cfg.ndf, base, cfg.n_up, spectral=cfg.spectral_norm).to(device)
    G.apply(dcgan_init)
    D.apply(dcgan_init)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    print(f"[{cfg.name}] {len(data)} images {cfg.size[0]}×{cfg.size[1]} | G {count_params(G):,} params "
          f"| D {count_params(D):,} params | n_dis {cfg.n_dis} | {device}")

    z_fixed = evaluation_latents(cfg.n_fixed, cfg.nz, device, seed=0)
    z_eval = evaluation_latents(n_eval, cfg.nz, device, seed=cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    aug = lambda t: diff_augment(t, cfg.diffaug)   # identity when the policy is empty

    def extra_real_batch(b: int) -> torch.Tensor:
        # Extra real batch drawn at random (additional D updates when n_dis > 1)
        x = to_model_range(data[torch.randint(len(data), (b,), device=data.device)])
        if cfg.flip:
            mask = torch.rand(b, device=x.device) < 0.5
            x[mask] = x[mask].flip(3)
        return x

    evals, best_fid, iters, t_start, start_epoch = [], float("inf"), 0, time.time(), 1
    hist_fields = ["epoch", "iters", "loss_d", "loss_g", "d_real", "d_fake", "logit_real", "logit_fake", "seconds"]
    resume_path = run / "checkpoints" / "resume.pt"

    if cfg.resume and resume_path.exists():
        # weights_only=False: the file also holds the state of the optimisers and of the random number
        # generators; it was written by this very script, so it can be trusted.
        # map_location="cpu": random generator states must stay on the CPU; weights and Adam states are
        # moved back to the GPU by load_state_dict.
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        diff = [k for k in RESUME_KEYS if st["config"].get(k) != asdict(cfg)[k]]
        if diff:
            raise ValueError(f"cannot resume, hyperparameters differ: {diff}")
        G.load_state_dict(st["G"]); D.load_state_dict(st["D"])
        opt_g.load_state_dict(st["opt_g"]); opt_d.load_state_dict(st["opt_d"])
        torch.set_rng_state(st["rng_cpu"])
        if st["rng_cuda"] is not None and device == "cuda":
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        gen.set_state(st["rng_batches"])
        evals, best_fid, iters = st["evals"], st["best_fid"], st["iters"]
        start_epoch = st["epoch"] + 1
        t_start = time.time() - st["elapsed"]          # the cumulative training time keeps running
        # history.csv may contain epochs later than the last saved state: drop them
        h = pd.read_csv(run / "history.csv")
        h[h["epoch"] <= st["epoch"]].to_csv(run / "history.csv", index=False)
        hist_f = open(run / "history.csv", "a", newline="", encoding="utf-8")
        hist = csv.DictWriter(hist_f, hist_fields)
        print(f"  resuming at epoch {start_epoch} (best FID so far: {best_fid:.1f})", flush=True)
    else:
        hist_f = open(run / "history.csv", "w", newline="", encoding="utf-8")
        hist = csv.DictWriter(hist_f, hist_fields)
        hist.writeheader()

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            t0 = time.time()
            acc = {"loss_d": 0.0, "loss_g": 0.0, "d_real": 0.0, "d_fake": 0.0, "logit_real": 0.0, "logit_fake": 0.0}
            n_batches = 0
            # One iteration = 1 G update, preceded by n_dis D updates. The number of G updates per epoch
            # is therefore the same whatever n_dis is (comparable budget across trials).
            for real in batches(data, cfg.batch_size, cfg.flip, gen):
                b = real.size(0)

                # ---- 1. Discriminator (n_dis times): tell real images from fake ones
                # With DiffAugment, D only ever sees augmented images — real and generated alike.
                for k in range(cfg.n_dis):
                    x = real if k == 0 else extra_real_batch(b)
                    with torch.no_grad():
                        fake_d = G(torch.randn(b, cfg.nz, device=device))
                    logit_real = D(aug(x))
                    logit_fake = D(aug(fake_d))     # generated without gradient: this step does not modify G
                    loss_d = d_loss(logit_real, logit_fake, cfg.loss, cfg.real_label)
                    opt_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    opt_d.step()

                # ---- 2. Generator: fool D (non-saturating loss in bce, −D(G(z)) in hinge)
                # The gradient flows through D (whose weights do not move: only opt_g takes a step) to G.
                # Generated images are augmented here too: the augmentation is differentiable, so the
                # gradient flows through it. This is what keeps the augmentations from "leaking" into G.
                fake = G(torch.randn(b, cfg.nz, device=device))
                loss_g = g_loss(D(aug(fake)), cfg.loss)
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()

                if not (torch.isfinite(loss_d) and torch.isfinite(loss_g)):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch} (D {loss_d.item()}, G {loss_g.item()})")

                # Tracking: D(x) and D(G(z)) are the "real" probabilities given by D (last D update).
                # At the theoretical equilibrium, both tend to 0.5.
                acc["loss_d"] += loss_d.item()
                acc["loss_g"] += loss_g.item()
                acc["d_real"] += torch.sigmoid(logit_real).mean().item()
                acc["d_fake"] += torch.sigmoid(logit_fake).mean().item()
                # Raw scores: the only meaningful reading with the hinge loss (expected margins: +1 / −1)
                acc["logit_real"] += logit_real.mean().item()
                acc["logit_fake"] += logit_fake.mean().item()
                n_batches += 1
                iters += 1

            hist.writerow({"epoch": epoch, "iters": iters, **{k: v / n_batches for k, v in acc.items()},
                           "seconds": time.time() - t0})
            hist_f.flush()

            if epoch % cfg.eval_every == 0 or epoch == cfg.epochs or epoch == 1:
                t_eval = time.time()
                save_image(generate(G, z_fixed).float() / 255, run / "samples" / f"epoch_{epoch:04d}.png", nrow=8, padding=2)
                feats, logits = mt.extract_features(generate(G, z_eval))
                scores = {"epoch": epoch, "iters": iters, "train_minutes": (time.time() - t_start) / 60,
                          **mt.evaluate(feats, logits, ref, light=cfg.eval_light)}
                evals.append(scores)
                pd.DataFrame(evals).to_csv(run / "eval.csv", index=False)
                if cfg.save_checkpoints:
                    state = {"G": G.state_dict(), "D": D.state_dict(), "config": asdict(cfg), "epoch": epoch, "scores": scores}
                    torch.save(state, run / "checkpoints" / "last.pt")
                    if scores["FID"] < best_fid:
                        torch.save(state, run / "checkpoints" / "best.pt")
                best_fid = min(best_fid, scores["FID"])
                if cfg.save_checkpoints and epoch < cfg.epochs:
                    # Full state for a resume (see the docstring); pointless at the very last epoch
                    save_resume_state(
                        resume_path, G=G.state_dict(), D=D.state_dict(), opt_g=opt_g.state_dict(),
                        opt_d=opt_d.state_dict(), config=asdict(cfg), epoch=epoch, iters=iters, evals=evals,
                        best_fid=best_fid, elapsed=time.time() - t_start, rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all() if device == "cuda" else None,
                        rng_batches=gen.get_state())
                ntest = f" (ntest {scores['FID_ntest']:.1f})" if "FID_ntest" in scores else ""
                print(f"  epoch {epoch:4d} | D {acc['loss_d'] / n_batches:.3f} G {acc['loss_g'] / n_batches:.3f} "
                      f"| FID {scores['FID']:.1f}{ntest} P {scores['precision']:.2f} "
                      f"R {scores['recall']:.2f} mem {scores['mem_ratio']:.2f} | eval {time.time() - t_eval:.0f} s", flush=True)
                if on_eval is not None:
                    on_eval(epoch, scores)
    finally:
        hist_f.close()

    resume_path.unlink(missing_ok=True)      # training finished: the (large) resume state is no longer needed
    if cfg.register:
        register(cfg, run, G, D, time.time() - t_start)
    return run


def register(cfg: Config, run: Path, G, D, seconds: float) -> None:
    """Add the best and the last checkpoint to the model comparison table."""
    ev = pd.read_csv(run / "eval.csv")
    rows = []
    for kind, r in [("best", ev.loc[ev["FID"].idxmin()]), ("last", ev.iloc[-1])]:
        rows.append({"run": cfg.name, "checkpoint": kind, "genre": cfg.genre, "size": f"{cfg.size[0]}x{cfg.size[1]}",
                     "loss": cfg.loss, "diffaug": cfg.diffaug or "-", "spectral_norm": cfg.spectral_norm,
                     "lr_g": cfg.lr_g, "lr_d": cfg.lr_d, "beta1": cfg.beta1, "beta2": cfg.beta2,
                     "batch_size": cfg.batch_size, "n_dis": cfg.n_dis,
                     "params_G": count_params(G), "params_D": count_params(D), "train_minutes": round(seconds / 60, 1),
                     **r.to_dict()})
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg.empty:
        reg = reg[reg["run"] != cfg.name]      # re-running a run replaces its rows
    pd.concat([reg, pd.DataFrame(rows)], ignore_index=True).to_csv(REGISTRY, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = Config()
    for f_name, f_def in asdict(defaults).items():
        if f_name == "size":
            continue
        if isinstance(f_def, bool):
            p.add_argument(f"--{f_name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f_def)
        else:
            p.add_argument(f"--{f_name.replace('_', '-')}", type=type(f_def), default=f_def)
    train(Config(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
