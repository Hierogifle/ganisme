"""DiffAugment — differentiable augmentation for GANs trained on little data.

Reference: Zhao, Liu, Lin, Zhu & Han (2020), "Differentiable Augmentation for Data-Efficient GAN
Training", NeurIPS. Commented re-implementation, adapted to rectangular images.

The problem: with a few thousand images, the discriminator ends up **memorising** the training set
(see notebook 04: D(x) → 0.98, D(G(z)) → 0.02). It no longer generalises and no longer gives a
useful signal to the generator.

Why not simply augment the real images? Because the generator would then learn to produce augmented
images (shifted, cut out, recoloured): the augmentations would "leak" into the generated images.

The DiffAugment solution: apply **the same family of random transformations to real AND fake
images**, at every pass through the discriminator, including during the generator update. The
discriminator never sees exactly the same real image twice, and since both distributions are
transformed in the same way, the goal of the generator remains to produce realistic *non-augmented*
images. The transformations are **differentiable** (additions, multiplications, index shifts): the
gradient flows through them to reach the generator.

Three families ("color,translation,cutout" policy recommended by the paper):
  - color       : random brightness, saturation and contrast;
  - translation : random shift of up to 1/8 of the size, borders filled with zeros (mid-grey in [-1, 1]);
  - cutout      : a rectangle half the size of the image set to zero, at a random position.
Each image of a batch gets its own random draws.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    """Apply the policy to a batch (N, C, H, W) of images in [-1, 1]."""
    if not policy:
        return x
    for name in policy.split(","):
        for fn in AUGMENTS[name]:
            x = fn(x)
    return x.contiguous()


# --------------------------------------------------------------------------- colour
def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    # Adds a constant in [-0.5, 0.5] to the whole image (lighter / darker).
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    # Moves each pixel away from or towards the mean of its channels (factor in [0, 2]):
    # 0 = greyscale, 1 = unchanged, 2 = colours twice as saturated.
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    # Moves each pixel away from or towards the mean of the image (factor in [0.5, 1.5]).
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + mean


# --------------------------------------------------------------------------- geometry
def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Shift each image by at most `ratio` of its height / width (independently on both axes).

    Implemented by indexing: the image is padded with a border of zeros, then each pixel is read at
    its shifted position. This is a plain index selection, hence differentiable with respect to the
    pixel values.
    """
    n, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ty = torch.randint(-sh, sh + 1, (n, 1, 1), device=x.device)
    tx = torch.randint(-sw, sw + 1, (n, 1, 1), device=x.device)
    gb, gy, gx = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(h, device=x.device),
                                torch.arange(w, device=x.device), indexing="ij")
    gy = torch.clamp(gy + ty + 1, 0, h + 1)
    gx = torch.clamp(gx + tx + 1, 0, w + 1)
    padded = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])            # 1-pixel border of zeros
    return padded.permute(0, 2, 3, 1).contiguous()[gb, gy, gx].permute(0, 3, 1, 2)


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Set to zero a rectangle of size `ratio` × (height, width), centred at random in the image.

    Forces the discriminator to judge the image on all of its parts, without being able to rely on a
    single memorised detail.
    """
    n, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    oy = torch.randint(0, h + (1 - ch % 2), (n, 1, 1), device=x.device)
    ox = torch.randint(0, w + (1 - cw % 2), (n, 1, 1), device=x.device)
    gb, gy, gx = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gy = torch.clamp(gy + oy - ch // 2, 0, h - 1)
    gx = torch.clamp(gx + ox - cw // 2, 0, w - 1)
    mask = torch.ones(n, h, w, dtype=x.dtype, device=x.device)
    mask[gb, gy, gx] = 0
    return x * mask.unsqueeze(1)


AUGMENTS = {
    "color": [rand_brightness, rand_saturation, rand_contrast],
    "translation": [rand_translation],
    "cutout": [rand_cutout],
}
