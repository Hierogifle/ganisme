"""DiffAugment — augmentation différentiable pour GAN entraînés sur peu de données.

Référence : Zhao, Liu, Lin, Zhu & Han (2020), « Differentiable Augmentation for Data-Efficient GAN
Training », NeurIPS. Réimplémentation commentée, adaptée aux images rectangulaires.

Le problème : avec quelques milliers d'images, le discriminateur finit par **mémoriser** le train
(cf. notebook 04 : D(x) → 0,98, D(G(z)) → 0,02). Il ne généralise plus et ne donne plus de signal
utile au générateur.

Pourquoi ne pas simplement augmenter les vraies images ? Parce que le générateur apprendrait alors à
produire des images augmentées (décalées, découpées, recolorées) : les augmentations « fuiraient »
dans les générations.

La solution de DiffAugment : appliquer **la même famille de transformations aléatoires aux vraies ET
aux fausses images**, à chaque passage dans le discriminateur, y compris pendant la mise à jour du
générateur. Le discriminateur ne voit jamais deux fois exactement la même image réelle, et comme les
deux distributions sont transformées de la même façon, l'objectif du générateur reste de produire
des images *non augmentées* réalistes. Les transformations sont **différentiables** (additions,
multiplications, décalages d'indices) : le gradient les traverse pour atteindre le générateur.

Trois familles (politique « color,translation,cutout » recommandée par l'article) :
  - color       : luminosité, saturation et contraste aléatoires ;
  - translation : décalage aléatoire jusqu'à 1/8 de la taille, bords remplis de zéros (gris moyen en [-1, 1]) ;
  - cutout      : un rectangle de la moitié de la taille de l'image mis à zéro, à une position aléatoire.
Chaque image d'un batch reçoit ses propres tirages aléatoires.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    """Applique la politique à un batch (N, C, H, W) d'images dans [-1, 1]."""
    if not policy:
        return x
    for name in policy.split(","):
        for fn in AUGMENTS[name]:
            x = fn(x)
    return x.contiguous()


# --------------------------------------------------------------------------- couleur
def rand_brightness(x: torch.Tensor) -> torch.Tensor:
    # Ajoute une constante dans [-0,5 ; 0,5] à toute l'image (plus claire / plus sombre).
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)


def rand_saturation(x: torch.Tensor) -> torch.Tensor:
    # Écarte ou rapproche chaque pixel de la moyenne de ses canaux (facteur dans [0 ; 2]) :
    # 0 = niveaux de gris, 1 = inchangé, 2 = couleurs deux fois plus saturées.
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + mean


def rand_contrast(x: torch.Tensor) -> torch.Tensor:
    # Écarte ou rapproche chaque pixel de la moyenne de l'image (facteur dans [0,5 ; 1,5]).
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + mean


# --------------------------------------------------------------------------- géométrie
def rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Décale chaque image d'au plus `ratio` de sa hauteur / largeur (indépendamment sur les deux axes).

    Implémentation par indexation : on remplit l'image d'une bordure de zéros, puis on lit chaque pixel
    à sa position décalée. C'est une simple sélection d'indices, donc différentiable par rapport aux
    valeurs des pixels.
    """
    n, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ty = torch.randint(-sh, sh + 1, (n, 1, 1), device=x.device)
    tx = torch.randint(-sw, sw + 1, (n, 1, 1), device=x.device)
    gb, gy, gx = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(h, device=x.device),
                                torch.arange(w, device=x.device), indexing="ij")
    gy = torch.clamp(gy + ty + 1, 0, h + 1)
    gx = torch.clamp(gx + tx + 1, 0, w + 1)
    padded = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])            # bordure de zéros d'1 pixel
    return padded.permute(0, 2, 3, 1).contiguous()[gb, gy, gx].permute(0, 3, 1, 2)


def rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Met à zéro un rectangle de taille `ratio` × (hauteur, largeur), centré au hasard dans l'image.

    Oblige le discriminateur à juger l'image sur l'ensemble de ses parties, sans pouvoir se reposer
    sur un seul détail mémorisé.
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
