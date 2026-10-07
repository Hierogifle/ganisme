"""GAN evaluation metrics — all computed in the Inception-v3 feature space.

Common principle: pixels are not compared (two very similar paintings can differ pixel by pixel);
**representations** extracted by a pre-trained network are compared instead. Each image — real or
generated — is converted by Inception-v3 into:
  - a vector of 2048 features (`pool3` layer): describes the visual content (textures, shapes, objects);
  - 1008 ImageNet classification logits: used only by the Inception Score.

The "TF-compatible" weights of torch-fidelity are used, identical to those of the original FID
implementation: scores are comparable with those published in the literature.

| Metric          | Measures                                 | Best     | Reference                          |
|-----------------|------------------------------------------|----------|------------------------------------|
| FID             | distance between distributions (quality + diversity) | ↓ 0 | Heusel et al., 2017             |
| KID             | same, without sample-size bias           | ↓ 0      | Bińkowski et al., 2018             |
| IS              | sharpness + variety of ImageNet classes  | ↑        | Salimans et al., 2016              |
| Precision       | share of generated images that look real | ↑ 1      | Kynkäänniemi et al., 2019          |
| Recall          | share of the real variety that is covered | ↑ 1     | Kynkäänniemi et al., 2019          |
| Density         | precision, robust to outliers            | ↑ ~1     | Naeem et al., 2020                 |
| Coverage        | recall, robust to outliers               | ↑ 1      | Naeem et al., 2020                 |
| Memorisation    | does the generator copy the training set? | ≈ 1     | nearest-neighbour test             |
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

_EXTRACTOR = None


# --------------------------------------------------------------------------- feature extraction
def get_extractor(device: str | None = None):
    """Inception-v3 (TF weights of the original FID), loaded only once."""
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _EXTRACTOR = FeatureExtractorInceptionV3("inception-v3-compat", ["2048", "logits_unbiased"]).to(device).eval()
        _EXTRACTOR.device = device
    return _EXTRACTOR


@torch.no_grad()
def extract_features(images: np.ndarray | torch.Tensor, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Features (N, 2048) and logits (N, 1008) of a set of images.

    `images`: uint8, either (N, H, W, 3) in numpy (dataset format) or (N, 3, H, W) in torch.
    The extractor resizes to 299×299 itself (same interpolation as TensorFlow): images are passed at
    their training resolution, real and generated alike.
    """
    fe = get_extractor()
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images).permute(0, 3, 1, 2)
    feats, logits = [], []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(fe.device)
        if batch.dtype != torch.uint8:
            raise TypeError("images must be uint8 in [0, 255]")
        f, lg = fe(batch)
        feats.append(f.double().cpu())
        logits.append(lg.double().cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).numpy()


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Generator output (tanh, in [-1, 1]) -> uint8 [0, 255], the format expected by the extractor."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


# --------------------------------------------------------------------------- FID / KID / IS
def fid(real: np.ndarray, fake: np.ndarray) -> float:
    """Fréchet Inception Distance.

    Each cloud of features is modelled by a Gaussian (mean μ, covariance Σ) and the Fréchet
    (Wasserstein-2) distance between the two Gaussians is computed:

        FID = ||μ_r − μ_f||²  +  Tr(Σ_r + Σ_f − 2 (Σ_r Σ_f)^½)

    The first term compares the "average content", the second the spread: a generator in mode collapse
    has a squashed covariance and a high FID, even if its images look good. Warning: FID is **biased by
    the sample size** (the smaller N, the higher the FID); only compare FIDs computed with the same N.
    """
    mu1, mu2 = real.mean(0), fake.mean(0)
    s1, s2 = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    # Tr((Σ_r Σ_f)^½) = sum of the square roots of the eigenvalues of Σ_r Σ_f (real and ≥ 0 in theory;
    # small imaginary / negative parts caused by numerical errors are discarded).
    # Equivalent to scipy.linalg.sqrtm but faster and more stable on 2048×2048 matrices.
    eig = linalg.eigvals(s1 @ s2)
    tr_covmean = np.sqrt(np.clip(eig.real, 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def kid(real: np.ndarray, fake: np.ndarray, n_subsets: int = 100, subset_size: int = 1000,
        seed: int = 0) -> tuple[float, float]:
    """Kernel Inception Distance: unbiased MMD² with a polynomial kernel k(x, y) = (x·y / d + 1)³.

    Unlike FID, its estimator is **unbiased**: it stays comparable across samples of different sizes,
    which makes it the right metric for small datasets (our test set holds a few hundred images). It is
    averaged over random subsamples.
    Returns (mean, standard deviation); it is usually displayed ×1000.
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
    """Inception Score: exp( E_x[ KL( p(y|x) || p(y) ) ] ).

    High if each image is classified with confidence (sharpness) AND if the classes are varied
    (diversity). **Important limitation for this project**: the classes are those of ImageNet (dogs,
    cars…), not painting categories; IS is therefore barely relevant for artworks and is only reported
    for information, for comparison with the literature.
    """
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    scores = []
    for part in np.array_split(p, splits):
        py = part.mean(0, keepdims=True)
        kl = (part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(1).mean()
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


# --------------------------------------------------------------------------- precision / recall
def _pairwise(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.cdist(torch.from_numpy(a).float().to(dev), torch.from_numpy(b).float().to(dev))


def _knn_radii(x: np.ndarray, k: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest neighbour (within its own cloud)."""
    d = _pairwise(x, x)
    return d.kthvalue(k + 1, dim=1).values  # k+1: the first neighbour is the point itself (distance 0)


def precision_recall_density_coverage(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict:
    """Four metrics that separate **quality** from **diversity**, which FID does not.

    The "manifold" of real images is approximated by the union of balls centred on each real image,
    with radius = distance to its k-th real neighbour (same for generated images).

    - Precision: share of generated images that fall inside the real manifold → *realism*.
    - Recall   : share of real images that fall inside the generated manifold → *diversity*.
      A mode collapse gives a high precision but a collapsed recall.
    - Density  : like precision, but counts how many real balls contain each generated image
      (normalised by k); ~1 for a good generator, less sensitive to real outliers.
    - Coverage : share of real images whose ball contains at least one generated image;
      a robust version of recall.
    """
    r_real = _knn_radii(real, k)
    r_fake = _knn_radii(fake, k)
    d_rf = _pairwise(real, fake)                          # (N_real, N_fake)
    in_real = d_rf <= r_real[:, None]                     # generated image j inside the ball of real image i
    precision = in_real.any(0).float().mean().item()
    recall = (d_rf <= r_fake[None, :]).any(1).float().mean().item()
    density = (in_real.sum(0).float() / k).mean().item()
    coverage = (d_rf.min(1).values <= r_real).float().mean().item()
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


# --------------------------------------------------------------------------- memorisation
def memorization(fake: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    """Does the generator copy its training images?

    For each generated image, the distance (in feature space) to the nearest **training** image is
    measured. It is compared with the same distance computed for real images **never seen** by the
    model (the test set): this is the "normal" gap between two different artworks of the same genre.

    - `mem_ratio` = median(d fake→train) / median(d test→train): ≈ 1 → normal; ≪ 1 → copies.
    - `copies`    = share of generated images closer to the training set than 99 % of the test images
      (threshold = 1st percentile of the test→train distances): ≈ 1 % expected by construction.
    """
    d_fake = _pairwise(fake, train).min(1).values.cpu().numpy()
    d_test = _pairwise(test, train).min(1).values.cpu().numpy()
    threshold = np.percentile(d_test, 1)
    return {"mem_ratio": float(np.median(d_fake) / np.median(d_test)),
            "copies": float((d_fake < threshold).mean())}


# --------------------------------------------------------------------------- full evaluation
@dataclass
class Reference:
    """Features of the real images of a genre at a given resolution (computed once, then cached)."""
    train: np.ndarray
    test: np.ndarray
    train_logits: np.ndarray
    test_logits: np.ndarray

    @classmethod
    def load_or_compute(cls, genre: str, size: tuple[int, int], cache_dir: Path) -> "Reference":
        from dataset import load_split  # local import: metrics.py stays usable without the dataset
        path = cache_dir / f"inception_ref_{size[0]}x{size[1]}.npz"
        if path.exists():
            z = np.load(path)
            return cls(z["train"], z["test"], z["train_logits"], z["test_logits"])
        tr, tr_lg = extract_features(load_split(genre, "train", size))
        te, te_lg = extract_features(load_split(genre, "test", size))
        np.savez_compressed(path, train=tr, test=te, train_logits=tr_lg, test_logits=te_lg)
        return cls(tr, te, tr_lg, te_lg)


def fid_at_n(real: np.ndarray, fake: np.ndarray, n: int, repeats: int = 3, seed: int = 0) -> float:
    """Mean FID over `repeats` subsamples of `n` generated images.

    Used to compare a model with the "floor" (the test set, which only holds a few hundred images)
    at equal sample size, since FID depends on the sample size.
    """
    if len(fake) <= n:
        return fid(real, fake)
    rng = np.random.default_rng(seed)
    return float(np.mean([fid(real, fake[rng.choice(len(fake), n, replace=False)]) for _ in range(repeats)]))


def evaluate(fake_feats: np.ndarray, fake_logits: np.ndarray, ref: Reference, k: int = 5,
             light: bool = False) -> dict:
    """Every metric for a set of generated images, against the training set of the genre.

    Protocol to follow when comparing models: same genre, same resolution, same number of generated
    images (see N_EVAL in notebook 03), same reference (the training set).

    Two FIDs are returned:
    - `FID`       : with all generated images (N_EVAL) → to compare models with each other;
    - `FID_ntest` : with as many images as the test set → directly comparable with the "unseen real"
                    floor, which can only be computed at that sample size.

    `light=True` (hyperparameter search): only FID, precision/recall/density/coverage and
    memorisation — about 8 s instead of 35 s. KID, FID_ntest and IS are skipped.
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
