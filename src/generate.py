"""Generate images from a trained model.

Usable from the command line or imported (notebooks, Streamlit application).

Examples:
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
    G.eval()   # BatchNorm in inference mode: a generated image does not depend on the others in the batch
    return G, {**cfg, "epoch": ckpt["epoch"], "scores": ckpt.get("scores", {})}


def load_generator(run: str, checkpoint: str = "best", device: str | None = None) -> tuple[Generator, dict]:
    """Reload the generator of a run (`best` = best FID, `last` = end of training)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(RUNS / run / "checkpoints" / f"{checkpoint}.pt", map_location=device, weights_only=True)
    return _build(ckpt, device)


def load_generator_file(path: str | Path, device: str | None = None) -> tuple[Generator, dict]:
    """Reload a generator exported by `src/export_models.py` (standalone file, no runs/ folder needed)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return _build(torch.load(path, map_location=device, weights_only=True), device)


def latents(n: int, nz: int, seed: int | None = None, truncation: float = 1.0) -> torch.Tensor:
    """n latent vectors z ~ N(0, I). Same seed = same vectors = same images.

    `truncation` < 1 pulls the vectors towards the origin: this is the truncation trick (Brock et al.,
    2019), meant to trade diversity for fidelity without retraining. **Tested on our models, it does
    not work**: from 0.8 the images get darker, and around 0.4 they melt into a dark mean image. It
    assumes a generator trained for it (BigGAN, StyleGAN); a DCGAN is not. The value used in training,
    1.0, is therefore kept; the application does not expose this setting.
    """
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return torch.randn(n, nz, generator=gen) * truncation


@torch.no_grad()
def render(G: Generator, z: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    """uint8 images (n, H, W, 3) from latent vectors."""
    device = next(G.parameters()).device
    out = [G(z[i:i + batch_size].to(device)).clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
           for i in range(0, len(z), batch_size)]
    return torch.cat(out).permute(0, 2, 3, 1).cpu().numpy()


def sample(G: Generator, n: int, seed: int | None = None, truncation: float = 1.0) -> np.ndarray:
    """n uint8 images (n, H, W, 3)."""
    return render(G, latents(n, G.nz, seed, truncation))


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation between two latent vectors.

    A straight-line interpolation would pass close to the origin, a region the generator almost never
    saw during training (high-dimensional Gaussian vectors all have roughly the same norm). Spherical
    interpolation stays at the right distance from the origin.
    """
    omega = torch.acos((a / a.norm() * b / b.norm()).sum().clamp(-1, 1))
    if omega.abs() < 1e-6:
        return a
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


def interpolate(G: Generator, seed_a: int, seed_b: int, steps: int = 8, truncation: float = 1.0) -> np.ndarray:
    """`steps` images going gradually from the image of seed A to the image of seed B."""
    a, b = latents(1, G.nz, seed_a, truncation)[0], latents(1, G.nz, seed_b, truncation)[0]
    return render(G, torch.stack([slerp(a, b, float(t)) for t in torch.linspace(0, 1, steps)]))


def to_grid(images: np.ndarray, ncol: int = 8, pad: int = 2, scale: int = 1) -> Image.Image:
    """Assemble images into one sheet; `scale` enlarges it (Lanczos) to see small images better."""
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
    src.add_argument("--run", help="name of a run in runs/")
    src.add_argument("--model", type=Path, help="exported file, e.g. models/portrait_128.pt")
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
    print(f"{args.n} images ({name}, epoch {info['epoch']}) -> {out}")


if __name__ == "__main__":
    main()
