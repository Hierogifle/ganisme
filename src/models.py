"""GAN architectures of the project.

DCGAN — Radford, Metz & Chintala (2015), "Unsupervised Representation Learning with Deep
Convolutional Generative Adversarial Networks". The stabilisation rules of the paper are followed:

  1. no pooling: down/up-sampling is learned by stride-2 convolutions;
  2. BatchNorm in both networks, except at the generator output and the discriminator input;
  3. no hidden dense layer: fully convolutional networks;
  4. generator: ReLU everywhere, Tanh at the output (images in [-1, 1]);
  5. discriminator: LeakyReLU(0.2) everywhere;
  6. weights initialised from N(0, 0.02).

Only adaptation: the images are not square. The generator starts from a `base` grid = (height,
width) instead of 4×4, then each layer doubles the size: for portraits, base (5, 4) and 4 doublings
give 80×64 (height × width), i.e. a 64×80 image with a 4:5 aspect ratio.

        z (100)                                       image 3×80×64
          │  ConvT kernel 5×4                              │  Conv 4×4 /2
     512 × 5 × 4                                      64 × 40 × 32
          │  ConvT 4×4 ×2                                  │  Conv 4×4 /2
     256 × 10 × 8                                    128 × 20 × 16
          │                                                │
     128 × 20 × 16                                   256 × 10 × 8
          │                                                │
      64 × 40 × 32                                   512 × 5 × 4
          │  ConvT 4×4 ×2 + Tanh                           │  Conv kernel 5×4
      3 × 80 × 64                                     1 logit (real / fake)
         GENERATOR                                      DISCRIMINATOR
"""
from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """Latent vector z (nz) -> image (3, base_h·2^n_up, base_w·2^n_up) in [-1, 1]."""

    def __init__(self, nz: int = 100, ngf: int = 64, base: tuple[int, int] = (5, 4), n_up: int = 4, nc: int = 3):
        super().__init__()
        self.nz = nz
        c = ngf * 2 ** (n_up - 1)                       # 512 channels on the starting grid
        layers = [
            # z is seen as a 1×1 "image" with nz channels; a transposed convolution whose kernel equals
            # the starting grid projects it to a c × base_h × base_w tensor.
            nn.ConvTranspose2d(nz, c, kernel_size=base, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
        ]
        for _ in range(n_up - 1):
            # Each block doubles the height and the width (kernel 4, stride 2, padding 1) and halves
            # the number of channels: "depth" is traded for resolution.
            layers += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        # Last layer: 3 RGB channels, no BatchNorm, Tanh to produce values in [-1, 1]
        # (same range as the normalised real images).
        layers += [nn.ConvTranspose2d(c, nc, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), self.nz, 1, 1))


class Discriminator(nn.Module):
    """Image -> one logit (real/fake score before the sigmoid). Mirror architecture of the generator.

    `spectral=True`: SNGAN variant (Miyato et al., 2018). Each convolution is wrapped in a **spectral
    normalisation**: at every forward pass its weights are divided by their largest singular value
    (estimated by one power-iteration step). Each layer then becomes 1-Lipschitz: a small change in the
    image can only produce a small change in the score. The discriminator can no longer react abruptly
    to changes in the generator — this targets the crises and collapses observed with model 2.
    BatchNorm is removed in this variant: it makes the output for one image depend on the other images
    of the batch, which breaks the Lipschitz guarantee. Without BatchNorm, the convolutions get a bias
    back.
    """

    def __init__(self, ndf: int = 64, base: tuple[int, int] = (5, 4), n_down: int = 4, nc: int = 3,
                 spectral: bool = False):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm if spectral else (lambda m: m)
        bias = spectral
        # First layer without BatchNorm (rule 2): the raw statistics of the image are let through.
        layers = [sn(nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)), nn.LeakyReLU(0.2, inplace=True)]
        c = ndf
        for _ in range(n_down - 1):
            # Each block halves the resolution and doubles the channels.
            # LeakyReLU rather than ReLU: a non-zero gradient for negative values, essential for the
            # generator to receive a signal even when the discriminator rejects its images.
            layers += [sn(nn.Conv2d(c, c * 2, 4, 2, 1, bias=bias))]
            if not spectral:
                layers += [nn.BatchNorm2d(c * 2)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            c *= 2
        # Final convolution with kernel = starting grid: summarises the whole image into a single score.
        # No sigmoid here: it is built into the loss (BCEWithLogits), which is numerically more stable.
        layers += [sn(nn.Conv2d(c, 1, kernel_size=base, stride=1, padding=0, bias=bias))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def dcgan_init(m: nn.Module) -> None:
    """Initialisation of the DCGAN paper: convolutions ~ N(0, 0.02), BatchNorm γ ~ N(1, 0.02), β = 0."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # With spectral normalisation, the "raw" weight is stored in parametrizations.weight.original
        w = m.parametrizations.weight.original if hasattr(m, "parametrizations") else m.weight
        nn.init.normal_(w, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def base_grid(size: tuple[int, int], n_up: int) -> tuple[int, int]:
    """Starting grid (height, width) for an image (width, height) after n_up doublings."""
    w, h = size
    f = 2 ** n_up
    if w % f or h % f:
        raise ValueError(f"{w}×{h} is not divisible by {f}")
    return h // f, w // f


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
