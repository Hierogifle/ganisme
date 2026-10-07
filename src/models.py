"""Architectures des GAN du projet.

DCGAN — Radford, Metz & Chintala (2015), « Unsupervised Representation Learning with Deep
Convolutional Generative Adversarial Networks ». On reprend les règles de stabilisation de l'article :

  1. pas de pooling : le sous/sur-échantillonnage est appris par des convolutions à stride 2 ;
  2. BatchNorm dans les deux réseaux, sauf en sortie du générateur et en entrée du discriminateur ;
  3. pas de couche dense cachée : réseaux entièrement convolutifs ;
  4. générateur : ReLU partout, Tanh en sortie (images dans [-1, 1]) ;
  5. discriminateur : LeakyReLU(0,2) partout ;
  6. poids initialisés selon N(0 ; 0,02).

Seule adaptation : les images ne sont pas carrées. Le générateur part d'une grille `base` = (hauteur,
largeur) au lieu de 4×4, puis chaque couche double la taille : pour les portraits, base (5, 4) et
4 doublements donnent 80×64 (hauteur × largeur), soit une image 64×80 au ratio 4:5.

        z (100)                                       image 3×80×64
          │  ConvT noyau 5×4                               │  Conv 4×4 /2
     512 × 5 × 4                                      64 × 40 × 32
          │  ConvT 4×4 ×2                                  │  Conv 4×4 /2
     256 × 10 × 8                                    128 × 20 × 16
          │                                                │
     128 × 20 × 16                                   256 × 10 × 8
          │                                                │
      64 × 40 × 32                                   512 × 5 × 4
          │  ConvT 4×4 ×2 + Tanh                           │  Conv noyau 5×4
      3 × 80 × 64                                     1 logit (réel / faux)
        GÉNÉRATEUR                                      DISCRIMINATEUR
"""
from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """Vecteur latent z (nz) -> image (3, base_h·2^n_up, base_w·2^n_up) dans [-1, 1]."""

    def __init__(self, nz: int = 100, ngf: int = 64, base: tuple[int, int] = (5, 4), n_up: int = 4, nc: int = 3):
        super().__init__()
        self.nz = nz
        c = ngf * 2 ** (n_up - 1)                       # 512 canaux sur la grille de départ
        layers = [
            # z est vu comme une « image » 1×1 à nz canaux ; une convolution transposée de noyau
            # égal à la grille de départ la projette en un tenseur c × base_h × base_w.
            nn.ConvTranspose2d(nz, c, kernel_size=base, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
        ]
        for _ in range(n_up - 1):
            # Chaque bloc double la hauteur et la largeur (noyau 4, stride 2, padding 1) et divise
            # le nombre de canaux par deux : on échange de la « profondeur » contre de la résolution.
            layers += [nn.ConvTranspose2d(c, c // 2, 4, 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        # Dernière couche : 3 canaux RGB, pas de BatchNorm, Tanh pour produire des valeurs dans [-1, 1]
        # (même échelle que les images réelles normalisées).
        layers += [nn.ConvTranspose2d(c, nc, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), self.nz, 1, 1))


class Discriminator(nn.Module):
    """Image -> un logit (score réel/faux avant sigmoïde). Architecture miroir du générateur.

    `spectral=True` : variante SNGAN (Miyato et al., 2018). Chaque convolution est enveloppée d'une
    **normalisation spectrale** : ses poids sont divisés à chaque passage par leur plus grande valeur
    singulière (estimée par une itération de la méthode de la puissance). Chaque couche devient alors
    1-lipschitzienne : une petite variation de l'image ne peut produire qu'une petite variation du score.
    Le discriminateur ne peut plus réagir brutalement aux changements du générateur — c'est ce qui
    vise les crises et effondrements observés avec le modèle n°2.
    La BatchNorm est retirée dans cette variante : elle fait dépendre la sortie d'une image des autres
    images du batch, ce qui casse la garantie de Lipschitz. Sans BatchNorm, les convolutions reprennent
    un biais.
    """

    def __init__(self, ndf: int = 64, base: tuple[int, int] = (5, 4), n_down: int = 4, nc: int = 3,
                 spectral: bool = False):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm if spectral else (lambda m: m)
        bias = spectral
        # 1re couche sans BatchNorm (règle 2) : on laisse passer les statistiques brutes de l'image.
        layers = [sn(nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias)), nn.LeakyReLU(0.2, inplace=True)]
        c = ndf
        for _ in range(n_down - 1):
            # Chaque bloc divise la résolution par deux et double les canaux.
            # LeakyReLU plutôt que ReLU : un gradient non nul pour les valeurs négatives, indispensable
            # pour que le générateur reçoive un signal même quand le discriminateur le rejette.
            layers += [sn(nn.Conv2d(c, c * 2, 4, 2, 1, bias=bias))]
            if not spectral:
                layers += [nn.BatchNorm2d(c * 2)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            c *= 2
        # Convolution finale de noyau = grille de départ : résume toute l'image en un seul score.
        # Pas de sigmoïde ici : elle est intégrée à la perte (BCEWithLogits), plus stable numériquement.
        layers += [sn(nn.Conv2d(c, 1, kernel_size=base, stride=1, padding=0, bias=bias))]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def dcgan_init(m: nn.Module) -> None:
    """Initialisation de l'article DCGAN : convolutions ~ N(0 ; 0,02), BatchNorm γ ~ N(1 ; 0,02), β = 0."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        # Avec la normalisation spectrale, le poids « brut » est stocké dans parametrizations.weight.original
        w = m.parametrizations.weight.original if hasattr(m, "parametrizations") else m.weight
        nn.init.normal_(w, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def base_grid(size: tuple[int, int], n_up: int) -> tuple[int, int]:
    """Grille de départ (hauteur, largeur) pour une image (largeur, hauteur) après n_up doublements."""
    w, h = size
    f = 2 ** n_up
    if w % f or h % f:
        raise ValueError(f"{w}×{h} n'est pas divisible par {f}")
    return h // f, w // f


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
