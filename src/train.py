"""Entraînement d'un GAN sur un genre, avec suivi des métriques.

À chaque itération, deux mises à jour alternées (cf. veille, question 6) :

  1. Discriminateur D — on lui montre un batch d'images réelles (cible 1) et un batch d'images
     générées (cible 0) ; on minimise l'entropie croisée binaire. Les images générées sont
     « détachées » du graphe : cette étape ne modifie pas le générateur.
  2. Générateur G — on génère un nouveau batch et on le passe dans D ; la cible est 1 (« fais croire
     à D que c'est vrai »). C'est la perte **non saturante** -log D(G(z)) : elle donne des gradients
     forts même quand D rejette facilement les faux, contrairement à log(1 - D(G(z))).

Toutes les `eval_every` époques :
  - une grille d'images générées à partir d'un bruit **fixe** (on voit le même « tableau » évoluer) ;
  - toutes les métriques de `metrics.evaluate` sur N_EVAL images (N_EVAL = taille du train) ;
  - un checkpoint ; le meilleur (FID le plus bas) est conservé à part.

Sorties : runs/<nom>/
    config.json      hyperparamètres
    history.csv      pertes et sorties de D, moyennées par époque
    eval.csv         métriques à chaque évaluation
    samples/         grilles d'images à bruit fixe
    checkpoints/     best.pt (meilleur FID) et last.pt
Et une ligne par modèle dans reports/metrics/models.csv (tableau comparatif du projet).

Exemple :
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
    res: int = 64                  # largeur ; la hauteur découle du ratio du genre
    epochs: int = 500
    batch_size: int = 128
    lr_g: float = 2e-4             # valeurs de l'article DCGAN
    lr_d: float = 2e-4
    beta1: float = 0.5             # β1 = 0,5 au lieu de 0,9 : moins d'inertie, plus stable pour les GAN
    beta2: float = 0.999
    nz: int = 100
    ngf: int = 64
    ndf: int = 64
    n_up: int = 4
    flip: bool = True              # augmentation : miroir horizontal aléatoire
    real_label: float = 1.0        # 0,9 = label smoothing (option, désactivée pour le DCGAN de base)
    diffaug: str = ""              # politique DiffAugment, ex. "color,translation,cutout" ("" = désactivé)
    loss: str = "bce"              # "bce" (DCGAN, non saturante) ou "hinge" (SNGAN)
    spectral_norm: bool = False    # normalisation spectrale dans D (remplace sa BatchNorm)
    n_dis: int = 1                 # mises à jour de D par mise à jour de G (5 dans l'article SNGAN)
    eval_every: int = 25
    eval_light: bool = False       # évaluation allégée (recherche d'hyperparamètres), cf. metrics.evaluate
    save_checkpoints: bool = True
    register: bool = True          # ajouter le run au tableau comparatif reports/metrics/models.csv
    resume: bool = False           # reprendre un entraînement interrompu (cf. checkpoints/resume.pt)
    n_fixed: int = 64              # images de la grille de suivi
    seed: int = 42
    size: tuple[int, int] = field(default=(0, 0))   # (largeur, hauteur), calculée

    def resolve(self) -> "Config":
        cfg = ds.GENRES[self.genre]
        self.size = (self.res, round(self.res / cfg.aspect))
        return self


# --------------------------------------------------------------------------- données
def load_train_tensor(cfg: Config, device: str) -> torch.Tensor:
    """Toutes les images d'entraînement en mémoire GPU (uint8, N×3×H×W) : 3 448 images de 64×80
    ne pèsent que 53 Mo, inutile de passer par un DataLoader qui relirait les fichiers à chaque époque."""
    x = ds.load_split(cfg.genre, "train", cfg.size)
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)


def to_model_range(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] -> float [-1, 1], l'échelle de sortie du Tanh du générateur."""
    return x_uint8.float().div_(127.5).sub_(1.0)


def batches(data: torch.Tensor, batch_size: int, flip: bool, gen: torch.Generator):
    """Mélange à chaque époque, dernier batch incomplet ignoré (BatchNorm n'aime pas les petits batchs)."""
    idx = torch.randperm(len(data), generator=gen, device="cpu").to(data.device)
    for i in range(0, len(idx) - batch_size + 1, batch_size):
        x = to_model_range(data[idx[i:i + batch_size]])
        if flip:
            mask = torch.rand(len(x), device=x.device) < 0.5
            x[mask] = x[mask].flip(3)       # miroir gauche-droite d'une image sur deux
        yield x


# --------------------------------------------------------------------------- génération / évaluation
@torch.no_grad()
def generate(G: Generator, z: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Images uint8 à partir de vecteurs latents. G passe en mode eval : BatchNorm utilise ses
    statistiques moyennes, comme ce sera le cas dans l'application (où l'on génère 1 image à la fois)."""
    was_training = G.training
    G.eval()
    out = torch.cat([mt.to_uint8(G(z[i:i + batch_size])) for i in range(0, len(z), batch_size)])
    G.train(was_training)
    return out


def evaluation_latents(n: int, nz: int, device: str, seed: int = 42) -> torch.Tensor:
    """Les MÊMES vecteurs latents pour toutes les évaluations et tous les modèles (protocole du notebook 03)."""
    return torch.randn(n, nz, generator=torch.Generator().manual_seed(seed)).to(device)


# --------------------------------------------------------------------------- pertes
def d_loss(logit_real: torch.Tensor, logit_fake: torch.Tensor, kind: str, real_label: float = 1.0) -> torch.Tensor:
    """Perte du discriminateur.

    - bce   : entropie croisée binaire, cibles 1 (réel) et 0 (faux) — la perte du GAN d'origine.
    - hinge : max(0, 1 − D(x)) + max(0, 1 + D(G(z))). D doit pousser les scores réels au-dessus de +1
              et les faux sous −1, puis **cesse d'être pénalisé** : il n'a aucun intérêt à devenir
              « infiniment sûr » de lui, ce qui limite sa domination (combiné à la normalisation spectrale).
    """
    if kind == "hinge":
        return F.relu(1 - logit_real).mean() + F.relu(1 + logit_fake).mean()
    return F.binary_cross_entropy_with_logits(logit_real, torch.full_like(logit_real, real_label)) + \
        F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))


def g_loss(logit_gen: torch.Tensor, kind: str) -> torch.Tensor:
    """Perte du générateur : bce non saturante −log D(G(z)), ou hinge −D(G(z)) (augmenter le score des faux)."""
    if kind == "hinge":
        return -logit_gen.mean()
    return F.binary_cross_entropy_with_logits(logit_gen, torch.ones_like(logit_gen))


# --------------------------------------------------------------------------- reprise
# Hyperparamètres qui doivent être identiques pour qu'une reprise ait un sens
RESUME_KEYS = ["genre", "res", "nz", "ngf", "ndf", "n_up", "loss", "spectral_norm", "lr_g", "lr_d", "beta1", "beta2",
               "n_dis", "diffaug", "batch_size", "flip", "real_label", "seed"]


def is_complete(cfg: Config) -> bool:
    """Vrai si le run a déjà atteint sa dernière époque et possède ses checkpoints."""
    run = RUNS / cfg.name
    if not (run / "eval.csv").exists() or not (run / "checkpoints" / "best.pt").exists():
        return False
    return int(pd.read_csv(run / "eval.csv")["epoch"].iloc[-1]) >= cfg.epochs


def save_resume_state(path: Path, **state) -> None:
    """Écrit l'état complet de l'entraînement. Écriture atomique : on écrit dans un fichier temporaire
    puis on le renomme, pour ne jamais laisser un fichier à moitié écrit si le PC s'arrête à ce moment."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------- entraînement
def train(cfg: Config, on_eval=None) -> Path:
    """Entraîne un GAN selon `cfg`.

    `on_eval(epoch, scores)` est appelé après chaque évaluation. Il peut lever une exception pour
    interrompre l'entraînement — c'est ainsi qu'Optuna arrête (« élague ») les essais peu prometteurs.

    Reprise (`cfg.resume`) : à chaque évaluation, l'état complet est sauvegardé dans
    `checkpoints/resume.pt` — poids des deux réseaux, état des deux optimiseurs Adam (leurs moyennes
    mobiles), générateurs aléatoires, historique des évaluations. Si l'entraînement est interrompu
    (veille, arrêt du PC, fermeture de la session), relancer la même commande repart de la dernière
    évaluation : on perd au plus `eval_every` époques. Un run déjà terminé n'est pas relancé.
    La reprise n'est pas identique au bit près à un entraînement ininterrompu (le GPU n'est pas
    déterministe), mais elle en est statistiquement équivalente.
    """
    cfg.resolve()
    run = RUNS / cfg.name
    if cfg.resume and is_complete(cfg):
        print(f"[{cfg.name}] déjà terminé ({cfg.epochs} époques), ignoré", flush=True)
        return run
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    (run / "samples").mkdir(parents=True, exist_ok=True)
    if cfg.save_checkpoints:
        (run / "checkpoints").mkdir(exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    # Données et référence des métriques
    data = load_train_tensor(cfg, device)
    ref = mt.Reference.load_or_compute(cfg.genre, cfg.size, ds.DATASETS / cfg.genre)
    n_eval = len(ref.train)

    # Modèles
    base = base_grid(cfg.size, cfg.n_up)
    G = Generator(cfg.nz, cfg.ngf, base, cfg.n_up).to(device)
    D = Discriminator(cfg.ndf, base, cfg.n_up, spectral=cfg.spectral_norm).to(device)
    G.apply(dcgan_init)
    D.apply(dcgan_init)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
    print(f"[{cfg.name}] {len(data)} images {cfg.size[0]}×{cfg.size[1]} | G {count_params(G):,} param. "
          f"| D {count_params(D):,} param. | n_dis {cfg.n_dis} | {device}")

    z_fixed = evaluation_latents(cfg.n_fixed, cfg.nz, device, seed=0)
    z_eval = evaluation_latents(n_eval, cfg.nz, device, seed=cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    aug = lambda t: diff_augment(t, cfg.diffaug)   # identité si la politique est vide

    def extra_real_batch(b: int) -> torch.Tensor:
        # Batch réel supplémentaire tiré au hasard (mises à jour additionnelles de D quand n_dis > 1)
        x = to_model_range(data[torch.randint(len(data), (b,), device=data.device)])
        if cfg.flip:
            mask = torch.rand(b, device=x.device) < 0.5
            x[mask] = x[mask].flip(3)
        return x

    evals, best_fid, iters, t_start, start_epoch = [], float("inf"), 0, time.time(), 1
    hist_fields = ["epoch", "iters", "loss_d", "loss_g", "d_real", "d_fake", "logit_real", "logit_fake", "seconds"]
    resume_path = run / "checkpoints" / "resume.pt"

    if cfg.resume and resume_path.exists():
        # weights_only=False : le fichier contient aussi l'état des optimiseurs et des générateurs aléatoires ;
        # il a été écrit par ce même script, on peut lui faire confiance.
        # map_location="cpu" : les états des générateurs aléatoires doivent rester sur le processeur ; les poids
        # et les états d'Adam sont replacés sur le GPU par load_state_dict.
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        diff = [k for k in RESUME_KEYS if st["config"].get(k) != asdict(cfg)[k]]
        if diff:
            raise ValueError(f"reprise impossible, hyperparamètres différents : {diff}")
        G.load_state_dict(st["G"]); D.load_state_dict(st["D"])
        opt_g.load_state_dict(st["opt_g"]); opt_d.load_state_dict(st["opt_d"])
        torch.set_rng_state(st["rng_cpu"])
        if st["rng_cuda"] is not None and device == "cuda":
            torch.cuda.set_rng_state_all(st["rng_cuda"])
        gen.set_state(st["rng_batches"])
        evals, best_fid, iters = st["evals"], st["best_fid"], st["iters"]
        start_epoch = st["epoch"] + 1
        t_start = time.time() - st["elapsed"]          # le temps d'entraînement cumulé continue de courir
        # history.csv peut contenir des époques postérieures à la dernière sauvegarde : on les retire
        h = pd.read_csv(run / "history.csv")
        h[h["epoch"] <= st["epoch"]].to_csv(run / "history.csv", index=False)
        hist_f = open(run / "history.csv", "a", newline="", encoding="utf-8")
        hist = csv.DictWriter(hist_f, hist_fields)
        print(f"  reprise à l'époque {start_epoch} (meilleur FID jusqu'ici : {best_fid:.1f})", flush=True)
    else:
        hist_f = open(run / "history.csv", "w", newline="", encoding="utf-8")
        hist = csv.DictWriter(hist_f, hist_fields)
        hist.writeheader()

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            t0 = time.time()
            acc = {"loss_d": 0.0, "loss_g": 0.0, "d_real": 0.0, "d_fake": 0.0, "logit_real": 0.0, "logit_fake": 0.0}
            n_batches = 0
            # Une itération = 1 mise à jour de G, précédée de n_dis mises à jour de D. Le nombre de mises à
            # jour de G par époque est donc le même quel que soit n_dis (budget comparable entre essais).
            for real in batches(data, cfg.batch_size, cfg.flip, gen):
                b = real.size(0)

                # ---- 1. Discriminateur (n_dis fois) : séparer les vraies images des fausses
                # Avec DiffAugment, D ne voit que des images augmentées — réelles comme générées.
                for k in range(cfg.n_dis):
                    x = real if k == 0 else extra_real_batch(b)
                    with torch.no_grad():
                        fake_d = G(torch.randn(b, cfg.nz, device=device))
                    logit_real = D(aug(x))
                    logit_fake = D(aug(fake_d))     # généré sans gradient : cette étape ne modifie pas G
                    loss_d = d_loss(logit_real, logit_fake, cfg.loss, cfg.real_label)
                    opt_d.zero_grad(set_to_none=True)
                    loss_d.backward()
                    opt_d.step()

                # ---- 2. Générateur : tromper D (perte non saturante en bce, −D(G(z)) en hinge)
                # Le gradient traverse D (dont les poids ne bougent pas : seul opt_g fait un pas) jusqu'à G.
                # Les images générées sont augmentées ici aussi : l'augmentation est différentiable,
                # le gradient la traverse. C'est ce qui empêche les augmentations de « fuir » dans G.
                fake = G(torch.randn(b, cfg.nz, device=device))
                loss_g = g_loss(D(aug(fake)), cfg.loss)
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                opt_g.step()

                if not (torch.isfinite(loss_d) and torch.isfinite(loss_g)):
                    raise FloatingPointError(f"perte non finie à l'époque {epoch} (D {loss_d.item()}, G {loss_g.item()})")

                # Suivi : D(x) et D(G(z)) sont les probabilités « réel » données par D (dernière mise à jour
                # de D). À l'équilibre théorique, les deux tendent vers 0,5.
                acc["loss_d"] += loss_d.item()
                acc["loss_g"] += loss_g.item()
                acc["d_real"] += torch.sigmoid(logit_real).mean().item()
                acc["d_fake"] += torch.sigmoid(logit_fake).mean().item()
                # Scores bruts : seule lecture valable avec la perte hinge (marges attendues : +1 / −1)
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
                    # État complet pour une reprise (cf. docstring) ; inutile à la toute dernière époque
                    save_resume_state(
                        resume_path, G=G.state_dict(), D=D.state_dict(), opt_g=opt_g.state_dict(),
                        opt_d=opt_d.state_dict(), config=asdict(cfg), epoch=epoch, iters=iters, evals=evals,
                        best_fid=best_fid, elapsed=time.time() - t_start, rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state_all() if device == "cuda" else None,
                        rng_batches=gen.get_state())
                ntest =f" (ntest {scores['FID_ntest']:.1f})" if "FID_ntest" in scores else ""
                print(f"  époque {epoch:4d} | D {acc['loss_d'] / n_batches:.3f} G {acc['loss_g'] / n_batches:.3f} "
                      f"| FID {scores['FID']:.1f}{ntest} P {scores['precision']:.2f} "
                      f"R {scores['recall']:.2f} mem {scores['mem_ratio']:.2f} | éval {time.time() - t_eval:.0f} s", flush=True)
                if on_eval is not None:
                    on_eval(epoch, scores)
    finally:
        hist_f.close()

    resume_path.unlink(missing_ok=True)      # entraînement terminé : l'état de reprise (volumineux) n'est plus utile
    if cfg.register:
        register(cfg, run, G, D, time.time() - t_start)
    return run


def register(cfg: Config, run: Path, G, D, seconds: float) -> None:
    """Ajoute le meilleur checkpoint et le dernier au tableau comparatif des modèles."""
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
        reg = reg[reg["run"] != cfg.name]      # relancer un run remplace ses lignes
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
