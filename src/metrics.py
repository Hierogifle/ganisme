"""Métriques d'évaluation des GAN — toutes calculées dans l'espace des features d'Inception-v3.

Principe commun : on ne compare pas les pixels (deux tableaux très semblables peuvent différer pixel à
pixel), mais des **représentations** extraites par un réseau pré-entraîné. Chaque image — réelle ou
générée — est convertie par Inception-v3 en :
  - un vecteur de 2048 features (couche `pool3`) : décrit le contenu visuel (textures, formes, objets) ;
  - 1008 logits de classification ImageNet : servent uniquement à l'Inception Score.

On utilise les poids « TF-compatibles » de torch-fidelity, identiques à ceux de l'implémentation
d'origine du FID : les scores sont comparables à ceux publiés dans la littérature.

| Métrique        | Mesure                                   | Meilleur | Référence                          |
|-----------------|------------------------------------------|----------|------------------------------------|
| FID             | distance entre distributions (qualité + diversité) | ↓ 0 | Heusel et al., 2017               |
| KID             | idem, sans biais de taille d'échantillon | ↓ 0      | Bińkowski et al., 2018             |
| IS              | netteté + variété des classes ImageNet   | ↑        | Salimans et al., 2016              |
| Précision       | part des images générées « réalistes »   | ↑ 1      | Kynkäänniemi et al., 2019          |
| Rappel          | part du réel couverte par le générateur  | ↑ 1      | Kynkäänniemi et al., 2019          |
| Densité         | précision robuste aux outliers           | ↑ ~1     | Naeem et al., 2020                 |
| Couverture      | rappel robuste aux outliers              | ↑ 1      | Naeem et al., 2020                 |
| Mémorisation    | le générateur recopie-t-il le train ?    | ≈ 1      | test du plus proche voisin         |
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

_EXTRACTOR = None


# --------------------------------------------------------------------------- extraction
def get_extractor(device: str | None = None):
    """Inception-v3 (poids TF du FID d'origine), chargé une seule fois."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _EXTRACTOR = FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"]).to(device).eval()
        _EXTRACTOR.device = device
    return _EXTRACTOR


@torch.no_grad()
def extract_features(images: np.ndarray | torch.Tensor, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Features (N, 2048) et logits (N, 1008) d'un lot d'images.

    `images` : uint8, soit (N, H, W, 3) en numpy (format du dataset), soit (N, 3, H, W) en torch.
    L'extracteur redimensionne lui-même en 299×299 (interpolation identique à TensorFlow) : on lui
    passe les images à leur résolution d'entraînement, réelles comme générées.
    """
    fe = get_extractor()
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
    feats, logits = [], []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(fe.device)
        if batch.dtype != torch.uint8:
            raise TypeError("les images doivent être en uint8 [0, 255]")
        f, lg = fe(batch)
        feats.append(f.double().cpu())
        logits.append(lg.double().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Sortie d'un générateur (tanh, dans [-1, 1]) -> uint8 [0, 255], format attendu par l'extracteur."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


# --------------------------------------------------------------------------- FID / KID / IS
def fid(real: np.ndarray, fake: np.ndarray) -> float:
    """Fréchet Inception Distance.

    On modélise chaque nuage de features par une gaussienne (moyenne μ, covariance Σ) et on calcule
    la distance de Fréchet (Wasserstein-2) entre les deux gaussiennes :

        FID = ||μ_r − μ_f||²  +  Tr(Σ_r + Σ_f − 2 (Σ_r Σ_f)^½)

    Le 1er terme compare le « contenu moyen », le 2nd la dispersion : un générateur en mode collapse a
    une covariance écrasée et un FID élevé, même si ses images sont belles. Attention : le FID est
    **biaisé par la taille d'échantillon** (plus N est petit, plus il est élevé) ; ne comparer que des
    FID calculés avec le même N.
    """
    mu1, mu2 = real.mean(0), fake.mean(0)
    s1, s2 = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    # Tr((Σ_r Σ_f)^½) = somme des racines des valeurs propres de Σ_r Σ_f (réelles et ≥ 0 en théorie ;
    # on écarte les petites parties imaginaires / négatives dues aux erreurs numériques).
    # Équivalent à scipy.linalg.sqrtm mais plus rapide et plus stable sur des matrices 2048×2048.
    eig = linalg.eigvals(s1 @ s2)
    tr_covmean = np.sqrt(np.clip(eig.real, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def kid(real: np.ndarray, fake: np.ndarray, n_subsets: int = 100, subset_size: int = 1000,
        seed: int = 0) -> tuple[float, float]:
    """Kernel Inception Distance : MMD² non biaisé avec un noyau polynomial k(x, y) = (x·y / d + 1)³.

    Contrairement au FID, son estimateur est **non biaisé** : il reste comparable entre des échantillons
    de tailles différentes, ce qui en fait la bonne métrique pour les petits jeux de données (notre test
    set fait quelques centaines d'images). On le moyenne sur des sous-échantillons tirés au hasard.
    Renvoie (moyenne, écart-type) ; on l'affiche souvent ×1000.
    """
    rng = np.random.default_rng(seed)
    d = real.shape[1]
    m = min(subset_size, len(real), len(fake))
    scores = []
    for _ in range(n_subsets):
        x = real[rng.choice(len(real), m, replace=False)]
        y = fake[rng.choice(len(fake), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (m * (m - 1)) - 2 * kxy.mean()
        scores.append(mmd)
    return float(np.mean(scores)), float(np.std(scores))


def inception_score(logits: np.ndarray, splits: int = 10) -> tuple[float, float]:
    """Inception Score : exp( E_x[ KL( p(y|x) || p(y) ) ] ).

    Élevé si chaque image est classée avec confiance (netteté) ET si les classes sont variées
    (diversité). **Limite importante pour ce projet** : les classes sont celles d'ImageNet (chiens,
    voitures…), pas des catégories de peinture ; l'IS est donc peu pertinent sur des œuvres d'art et
    n'est fourni qu'à titre indicatif, pour comparaison avec la littérature.
    """
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    scores = []
    for part in np.array_split(p, splits):
        py = part.mean(0, keepdims=True)
        kl = (part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(1).mean()
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


# --------------------------------------------------------------------------- précision / rappel
def _pairwise(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.cdist(torch.from_numpy(a).float().to(dev), torch.from_numpy(b).float().to(dev))


def _knn_radii(x: np.ndarray, k: int) -> torch.Tensor:
    """Distance de chaque point à son k-ième plus proche voisin (dans son propre nuage)."""
    d = _pairwise(x, x)
    return d.kthvalue(k + 1, dim=1).values  # k+1 : le 1er voisin est le point lui-même (distance 0)


def precision_recall_density_coverage(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict:
    """Quatre métriques qui séparent **qualité** et **diversité**, ce que le FID ne fait pas.

    On approxime la « variété » des images réelles par l'union de boules centrées sur chaque image
    réelle, de rayon = distance à son k-ième voisin réel (idem pour les images générées).

    - Précision : part des images générées qui tombent dans la variété réelle → *réalisme*.
    - Rappel    : part des images réelles qui tombent dans la variété générée → *diversité*.
      Un mode collapse donne une précision élevée mais un rappel effondré.
    - Densité   : comme la précision, mais compte combien de boules réelles contiennent chaque image
      générée (normalisé par k) ; ~1 pour un bon générateur, moins sensible aux outliers réels.
    - Couverture : part des images réelles dont la boule contient au moins une image générée ;
      version robuste du rappel.
    """
    r_real = _knn_radii(real, k)
    r_fake = _knn_radii(fake, k)
    d_rf = _pairwise(real, fake)                          # (N_real, N_fake)
    in_real = d_rf <= r_real[:, None]                     # image générée j dans la boule du réel i
    precision = in_real.any(0).float().mean().item()
    recall = (d_rf <= r_fake[None, :]).any(1).float().mean().item()
    density = (in_real.sum(0).float() / k).mean().item()
    coverage = (d_rf.min(1).values <= r_real).float().mean().item()
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


# --------------------------------------------------------------------------- mémorisation
def memorization(fake: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    """Le générateur recopie-t-il ses images d'entraînement ?

    Pour chaque image générée, on mesure la distance (features) à l'image du **train** la plus proche.
    On la compare à la même distance calculée pour des images réelles **jamais vues** (le test set) :
    c'est l'écart « normal » entre deux œuvres différentes du même genre.

    - `mem_ratio` = médiane(d fake→train) / médiane(d test→train) : ≈ 1 → normal ; ≪ 1 → copies.
    - `copies`    = part des images générées plus proches du train que 99 % des images test
      (seuil = 1er percentile des distances test→train) : ≈ 1 % attendu par construction.
    """
    d_fake = _pairwise(fake, train).min(1).values.cpu().numpy()
    d_test = _pairwise(test, train).min(1).values.cpu().numpy()
    threshold = np.percentile(d_test, 1)
    return {"mem_ratio": float(np.median(d_fake) / np.median(d_test)),
            "copies": float((d_fake < threshold).mean())}


# --------------------------------------------------------------------------- évaluation complète
@dataclass
class Reference:
    """Features des images réelles d'un genre à une résolution donnée (calculées une fois, en cache)."""
    train: np.ndarray
    test: np.ndarray
    train_logits: np.ndarray
    test_logits: np.ndarray

    @classmethod
    def load_or_compute(cls, genre: str, size: tuple[int, int], cache_dir: Path) -> "Reference":
        from dataset import load_split  # import local : metrics.py reste utilisable sans le dataset
        path = cache_dir / f"inception_ref_{size[0]}x{size[1]}.npz"
        if path.exists():
            z = np.load(path)
            return cls(z["train"], z["test"], z["train_logits"], z["test_logits"])
        tr, tr_lg = extract_features(load_split(genre, "train", size))
        te, te_lg = extract_features(load_split(genre, "test", size))
        np.savez_compressed(path, train=tr, test=te, train_logits=tr_lg, test_logits=te_lg)
        return cls(tr, te, tr_lg, te_lg)


def fid_at_n(real: np.ndarray, fake: np.ndarray, n: int, repeats: int = 3, seed: int = 0) -> float:
    """FID moyen sur `repeats` sous-échantillons de `n` images générées.

    Sert à comparer un modèle au « plancher » (le test set, qui ne compte que quelques centaines
    d'images) à effectif égal, puisque le FID dépend de la taille d'échantillon.
    """
    if len(fake) <= n:
        return fid(real, fake)
    rng = np.random.default_rng(seed)
    return float(np.mean([fid(real, fake[rng.choice(len(fake), n, replace=False)]) for _ in range(repeats)]))


def evaluate(fake_feats: np.ndarray, fake_logits: np.ndarray, ref: Reference, k: int = 5,
             light: bool = False) -> dict:
    """Toutes les métriques d'un lot d'images générées, par rapport au train set du genre.

    Protocole à respecter pour comparer des modèles : même genre, même résolution, même nombre d'images
    générées (voir N_EVAL dans le notebook 03), même référence (le train set).

    Deux FID sont renvoyés :
    - `FID`       : avec toutes les images générées (N_EVAL) → pour comparer les modèles entre eux ;
    - `FID_ntest` : avec autant d'images que le test set → directement comparable au plancher
                    « réel non vu », qui ne peut être calculé qu'avec cet effectif.

    `light=True` (recherche d'hyperparamètres) : seulement FID, précision/rappel/densité/couverture et
    mémorisation — environ 8 s au lieu de 35 s. KID, FID_ntest et IS sont omis.
    """
    base = {"n": len(fake_feats), "FID": fid(ref.train, fake_feats),
            **precision_recall_density_coverage(ref.train, fake_feats, k=k),
            **memorization(fake_feats, ref.train, ref.test)}
    if light:
        return base
    kid_m, kid_s = kid(ref.train, fake_feats)
    is_m, is_s = inception_score(fake_logits)
    return {
        "n": len(fake_feats),
        "FID": fid(ref.train, fake_feats),
        "FID_ntest": fid_at_n(ref.train, fake_feats, len(ref.test)),
        "KID_x1000": kid_m * 1000, "KID_std_x1000": kid_s * 1000,
        "IS": is_m, "IS_std": is_s,
        **precision_recall_density_coverage(ref.train, fake_feats, k=k),
        **memorization(fake_feats, ref.train, ref.test),
    }
