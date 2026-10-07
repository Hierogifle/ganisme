"""Générer des images à partir d'un modèle entraîné.

Utilisable en ligne de commande ou importé (notebooks, application Streamlit).

Exemples :
    python src/generate.py --run dcgan_diffaug_portrait_64 --n 32 --out outputs/portraits.png
    python src/generate.py --model models/portrait_128.pt --n 16 --seed 7 --scale 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import Generator, base_grid

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def _build(ckpt: dict, device: str) -> tuple[Generator, dict]:
    cfg = ckpt["config"]
    G = Generator(cfg["nz"], cfg["ngf"], base_grid(tuple(cfg["size"]), cfg["n_up"]), cfg["n_up"]).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()   # BatchNorm en mode inférence : une image générée ne dépend pas des autres du batch
    return G, {**cfg, "epoch": ckpt["epoch"], "scores": ckpt.get("scores", {})}


def load_generator(run: str, checkpoint: str = "best", device: str | None = None) -> tuple[Generator, dict]:
    """Recharge le générateur d'un run (`best` = meilleur FID, `last` = fin d'entraînement)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(RUNS / run / "checkpoints" / f"{checkpoint}.pt", map_location=device, weights_only=True)
    return _build(ckpt, device)


def load_generator_file(path: str | Path, device: str | None = None) -> tuple[Generator, dict]:
    """Recharge un générateur exporté par `src/export_models.py` (fichier autonome, sans le dossier runs/)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return _build(torch.load(path, map_location=device, weights_only=True), device)


def latents(n: int, nz: int, seed: int | None = None, truncation: float = 1.0) -> torch.Tensor:
    """n vecteurs latents z ~ N(0, I). Même graine = mêmes vecteurs = mêmes images.

    `truncation` < 1 rapproche les vecteurs de l'origine : c'est l'astuce de troncature (Brock et al., 2019),
    censée échanger de la diversité contre de la fidélité sans réentraîner. **Testée sur nos modèles, elle ne
    fonctionne pas** : dès 0,8 les images s'assombrissent, et vers 0,4 elles se fondent en une image moyenne
    sombre. Elle suppose un générateur entraîné pour (BigGAN, StyleGAN) ; un DCGAN ne l'est pas. On garde donc
    1.0, la valeur de l'entraînement ; l'application n'expose pas ce réglage.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(n, nz, generator=gen) * truncation


@torch.no_grad()
def render(G: Generator, z: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    """Images uint8 (n, H, W, 3) à partir de vecteurs latents."""
    device = next(G.parameters()).device
    out = [G(z[i:i + batch_size].to(device)).clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
           for i in range(0, len(z), batch_size)]
    return torch.cat(out).permute(0, 2, 3, 1).cpu().numpy()


def sample(G: Generator, n: int, seed: int | None = None, truncation: float = 1.0) -> np.ndarray:
    """n images uint8 (n, H, W, 3)."""
    return render(G, latents(n, G.nz, seed, truncation))


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Interpolation sphérique entre deux vecteurs latents.

    Une interpolation en ligne droite passerait près de l'origine, une zone que le générateur n'a presque
    jamais vue à l'entraînement (les vecteurs gaussiens de grande dimension ont tous à peu près la même
    norme). L'interpolation sphérique reste à la bonne distance de l'origine.
    """
    omega = torch.acos((a / a.norm() * b / b.norm()).sum().clamp(-1, 1))
    if omega.abs() < 1e-6:
        return a
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


def interpolate(G: Generator, seed_a: int, seed_b: int, steps: int = 8, truncation: float = 1.0) -> np.ndarray:
    """`steps` images allant progressivement de l'image de la graine A à celle de la graine B."""
    a, b = latents(1, G.nz, seed_a, truncation)[0], latents(1, G.nz, seed_b, truncation)[0]
    return render(G, torch.stack([slerp(a, b, float(t)) for t in torch.linspace(0, 1, steps)]))


def to_grid(images: np.ndarray, ncol: int = 8, pad: int = 2, scale: int = 1) -> Image.Image:
    """Assemble des images en une planche ; `scale` agrandit (Lanczos) pour mieux voir de petites images."""
    n, h, w, _ = images.shape
    nrow = int(np.ceil(n / ncol))
    grid = np.full((nrow * (h + pad) + pad, ncol * (w + pad) + pad, 3), 255, dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, ncol)
        grid[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = im
    out = Image.fromarray(grid)
    return out.resize((out.width * scale, out.height * scale), Image.LANCZOS) if scale > 1 else out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="nom d'un run dans runs/")
    src.add_argument("--model", type=Path, help="fichier exporté, ex. models/portrait_128.pt")
    p.add_argument("--checkpoint", default="best", choices=["best", "last"])
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--truncation", type=float, default=1.0)
    p.add_argument("--ncol", type=int, default=8)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    G, info = load_generator(args.run, args.checkpoint) if args.run else load_generator_file(args.model)
    name = args.run or args.model.stem
    out = args.out or ROOT / "outputs" / f"{name}_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    to_grid(sample(G, args.n, args.seed, args.truncation), args.ncol, scale=args.scale).save(out)
    print(f"{args.n} images ({name}, époque {info['epoch']}) -> {out}")


if __name__ == "__main__":
    main()
