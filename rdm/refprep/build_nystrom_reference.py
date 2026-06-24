"""Build the frozen per-encoder Nystrom reference bundle from a raw-feature pool.

Driver around :func:`rdm.compare.reference_math.build_nystrom_reference`: load the raw
``(N, d)`` pool (from :mod:`rdm.refprep.extract_features`), read the per-encoder median
bandwidth (optionally scaled by ``sigma_scale`` for the cold / joint kernel), build the
``m=4096`` k-means landmarks + ``alpha = K_ZZ^{-1} mu_bar`` + the ``k_rr`` constant, and save
the ``{Z, alpha, sigma, k_rr}`` bundle the training loss consumes.
"""
from ..compare.reference_math import build_nystrom_reference
from ..utils.io import load_bundle, save_bundle


def build_one(pool_path: str, sigma_path: str, out_path: str, n_landmarks: int = 4096,
              sigma_scale: float = 1.0, fit_n: int = 200000, kmeans_iters: int = 20,
              krr_n: int = 100000, seed: int = 0) -> dict:
    """Build + save one encoder's Nystrom bundle.

    Args:
        pool_path: ``{"features": (N,d)}`` raw-feature pool.
        sigma_path: ``{"sigma": float}`` median bandwidth (e.g. from ``cache_median_sigma``).
        out_path: destination ``.pt``.
        sigma_scale: multiply the read bandwidth (``0.25`` for the cold image kernel).
    """
    feats = load_bundle(pool_path)["features"].float()
    sigma = float(load_bundle(sigma_path)["sigma"]) * float(sigma_scale)
    bundle = build_nystrom_reference(feats, sigma, n_landmarks=n_landmarks, fit_n=fit_n,
                                     kmeans_iters=kmeans_iters, krr_n=krr_n, seed=seed)
    bundle["sigma_scale"] = float(sigma_scale)
    bundle["pool_path"] = pool_path
    save_bundle(bundle, out_path)
    return bundle
