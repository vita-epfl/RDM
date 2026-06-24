"""Nystrom kernel-mean attraction toward the frozen full-data reference.

The attraction term of the iRDM loss compares the generated batch against the *whole*
training set. Resampling the reference each step would inject noise that grows as the
bandwidth shrinks, so the data side is compressed once into a Nystrom kernel mean
embedding (Chatalic et al., 2022) over ``m = 4096`` k-means landmarks and frozen.

Two equivalent forms live here:

* :func:`nystrom_attraction` -- the **production** form used in training. The
  reference is the precomputed coefficient vector ``alpha = K_ZZ^{-1} mu_bar``
  (see :func:`rdm.compare.reference_math.build_nystrom_reference`), so the attraction
  is ``k_gr ~= mean_i k(g_i, Z) . alpha`` at cost ``O(B m)``.
* :func:`nystrom_feature_map` -- the **explicit** form in the paper (eq. 3),
  ``psi(x) = K_ZZ^{-1/2} (k(x, l_1), ..., k(x, l_m))^T`` via a symmetric
  eigendecomposition, with ``mu_bar = mean_t psi(r_t)``. Used by the toy diagnostic.

The two are algebraically identical for the cross term:
``psi(g)^T mu_bar = k(g, Z) K_ZZ^{-1} mu_Zr = k(g, Z) . alpha``.
"""
import torch

from .kernels import gaussian_gram, squared_distance


def nystrom_attraction(g: torch.Tensor, Z: torch.Tensor, Z2: torch.Tensor,
                       alpha: torch.Tensor, gamma: float) -> torch.Tensor:
    """Nystrom attraction (cross-kernel mean) ``k_gr = mean_i k(g_i, Z) . alpha``.

    Args:
        g: ``(N_g, d)`` generated features (carry grad).
        Z: ``(M, d)`` frozen landmarks; ``Z2 = (Z*Z).sum(1)``; ``alpha`` ``(M,)`` the
            precomputed ``K_ZZ^{-1} mu_bar`` coefficients. All frozen (fp32).
        gamma: RBF precision ``1/(2 sigma^2)``.
    """
    d2 = (g.pow(2).sum(1)[:, None] + Z2[None, :] - 2.0 * (g @ Z.T)).clamp_min(0)
    return (torch.exp(-gamma * d2) @ alpha).mean()


def nystrom_feature_map(Z: torch.Tensor, gamma: float, jitter: float = 1e-6):
    """Explicit Nystrom feature map ``psi(x) = K_ZZ^{-1/2} k(x, Z)`` (paper eq. 3 form).

    Returns a callable ``psi(x) -> (N, M)`` built from a symmetric eigendecomposition of
    ``K_ZZ`` with clamped eigenvalues. Equivalent to :func:`nystrom_attraction` once the
    reference mean ``mu_bar = psi(reference).mean(0)`` is formed: this is the form used
    by the spiral toy (:mod:`rdm.toy`); production training uses the precomputed-alpha
    form for a frozen, zero-variance reference.
    """
    M = Z.shape[0]
    K_ZZ = gaussian_gram(Z, Z, gamma) + jitter * torch.eye(M, device=Z.device, dtype=Z.dtype)
    evals, evecs = torch.linalg.eigh(K_ZZ)
    whiten = evecs @ torch.diag(evals.clamp_min(jitter).rsqrt())  # K_ZZ^{-1/2}

    def psi(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-gamma * squared_distance(x, Z)) @ whiten

    return psi
