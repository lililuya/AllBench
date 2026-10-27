"""Metrics from LyCORIS §5.2 / Appendix D.4; no cross-metric composite score."""
import numpy as np

from .io import require


def matrix(value):
    x = np.asarray(value, dtype=np.float64)
    require(x.ndim == 2 and min(x.shape) > 0 and np.isfinite(x).all(), "INVALID_FEATURE_MATRIX")
    return x


def unit(value):
    x = matrix(value)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    require((norms > 0).all(), "ZERO_FEATURE_VECTOR")
    return x / norms


def image_similarity(reference, generated):
    a, b = unit(reference), unit(generated)
    require(a.shape[1] == b.shape[1], "FEATURE_DIMENSION_MISMATCH")
    # All cross pairs, computed via means without allocating O(n*m) memory.
    ma, mb = a.mean(0), b.mean(0)
    return {"cosine": float(ma @ mb), "squared_centroid_distance": float(np.square(ma - mb).sum())}


def text_alignment(text, images):
    a, b = unit(text), unit(images)
    require(a.shape == b.shape, "PAIRED_FEATURE_SHAPE_MISMATCH")
    return float(np.einsum("ij,ij->i", a, b).mean())


def vendi(features):
    x = unit(features)
    require(len(x) >= 2, "DIVERSITY_REQUIRES_TWO_SEEDS")
    # Non-zero eigenvalues of XX'/n and X'X/n are identical.
    gram = x @ x.T if len(x) <= x.shape[1] else x.T @ x
    eigen = np.linalg.eigvalsh(gram / len(x))
    require(eigen.min() >= -1e-10, "NON_PSD_KERNEL")
    eigen = eigen[eigen > 1e-14]
    return float(np.exp(-np.sum(eigen * np.log(eigen))))


def gram_features(feature_maps):
    result = []
    for f in feature_maps:
        x = np.asarray(f, dtype=np.float64)
        require(x.ndim == 3 and min(x.shape) > 0 and np.isfinite(x).all(), "INVALID_CONV_FEATURE")
        c, h, w = x.shape
        x = x.reshape(c, h * w)
        result.append((x @ x.T / (c * h * w)).ravel())
    require(bool(result), "EMPTY_GRAM_LAYERS")
    return np.concatenate(result)


def paired_style(base, adapted):
    a, b = matrix(base), matrix(adapted)
    require(a.shape == b.shape, "PAIRED_STYLE_SHAPE_MISMATCH")
    return float(np.square(a - b).sum(1).mean())


def reference_style(reference, generated):
    a, b = matrix(reference), matrix(generated)
    require(a.shape[1] == b.shape[1], "FEATURE_DIMENSION_MISMATCH")
    # Mean squared distance over every cross pair, without a pairwise tensor.
    value = np.square(a).sum(1).mean() + np.square(b).sum(1).mean() - 2 * (a.mean(0) @ b.mean(0))
    return float(max(0.0, value))
