"""Build the joint (image-text) Nystrom reference for the conditional objective.

The joint coordinate is the concatenation ``[phi(x) | beta * tau(c)]`` over a paired pool
(COCO images + their captions, optionally mixed with teacher FLUX generations), compared
under a single Gaussian kernel at the cold image bandwidth ``sigma_img = sigma_scale *
median``. The text scale ``s_txt`` sets the text kernel bandwidth via ``beta = sigma_img /
s_txt`` (text weighted at 1 when ``s_txt`` is the text median). The widened feature then
goes through the same :func:`rdm.compare.reference_math.build_nystrom_reference`.

``phi`` (image features) and ``tau`` (the frozen L2-normalized text table) must be
row-aligned to the same caption ordering (:mod:`rdm.data.coco`).
"""
import torch

from ..compare.median_heuristic import median_bandwidth
from ..compare.reference_math import build_nystrom_reference
from ..utils.io import load_bundle, save_bundle


def build_joint_one(img_pool_path: str, tau_path: str, sigma_path: str, out_path: str,
                    sigma_scale: float = 0.25, s_txt: float | None = None,
                    n_landmarks: int = 4096, **kw) -> dict:
    """Build + save one encoder's joint Nystrom bundle over ``[phi | beta*tau]``.

    Args:
        img_pool_path: ``{"features": (N, d_img)}`` image features (row-aligned to captions).
        tau_path: frozen text table ``(N, d_txt)`` (.npy) -- the L2-normalized tau(c).
        sigma_path: ``{"sigma"}`` image median bandwidth (pre-scale).
        sigma_scale: cold image-kernel multiplier (0.25).
        s_txt: text scale; if ``None`` use the text median (-> text weight 1).
    """
    import numpy as np
    phi = load_bundle(img_pool_path)["features"].float()
    tau = torch.from_numpy(np.ascontiguousarray(np.load(tau_path))).float()
    assert tau.shape[0] == phi.shape[0], "image / text rows must be aligned"
    sigma_img = float(load_bundle(sigma_path)["sigma"]) * float(sigma_scale)
    if s_txt is None:
        s_txt = median_bandwidth(tau)
    beta = sigma_img / float(s_txt)
    joint = torch.cat([phi, beta * tau], dim=1)                 # [phi | beta*tau]
    bundle = build_nystrom_reference(joint, sigma_img, n_landmarks=n_landmarks, **kw)
    bundle.update(sigma_scale=float(sigma_scale), s_txt=float(s_txt), beta=float(beta),
                  d_img=int(phi.shape[1]), d_txt=int(tau.shape[1]))
    save_bundle(bundle, out_path)
    return bundle
