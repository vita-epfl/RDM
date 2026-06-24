"""Coupled image-text feature for the conditional (text-to-image) objective.

On a prompted task we match the *joint* law of image and caption features rather than
the image marginal (eq. 4). With a frozen text encoder ``tau`` (a SigLIP text tower; see
:mod:`rdm.representation.text_encoder`) the coupled feature is the concatenation

    Phi(x, c) = [ phi(x) | beta * tau(c) ]

compared under a single Gaussian kernel. Because the kernel of a concatenation factorizes,
``k(Phi, Phi') = k_img(phi, phi'; sigma_img) * k_txt(tau, tau'; sigma_img / beta)``, the
text weight is set by ``beta``: with the per-encoder image bandwidth ``sigma_img`` and a
text scale ``s_txt``, ``beta = sigma_img / s_txt`` makes the text kernel bandwidth ``s_txt``
(the paper weights the text component at 1). The image bandwidth itself is the cold
``0.25 * median`` for the joint runs.

The text embeddings ``tau(c)`` are precomputed once and frozen (``requires_grad=False``), so
the appended columns add zero parameter gradient -- gradient flows only through ``phi(x)``.
The real pairs couple each image with its caption; the generated pairs couple each output
with the prompt that produced it. Both must build ``Phi`` identically (same caption index,
same ``beta``, same frozen ``tau`` table) for the gradient-caching passes to agree, and the
frozen Nystrom landmarks must have been fit on the same ``[phi | beta*tau]`` ordering.
"""
import torch

#: The joint image bandwidth is the cold 0.25 * median heuristic.
JOINT_BANDWIDTH_SCALE = 0.25


def text_weight_beta(sigma_img: float, s_txt: float = 1.0) -> float:
    """``beta = sigma_img / s_txt`` -- the text-block scale that sets the text kernel bandwidth."""
    return float(sigma_img) / float(s_txt)


def gather_text(tau_table: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
    """Look up the frozen per-caption text embeddings ``tau(c)`` for a batch of caption ids."""
    return tau_table[caption_ids.to(tau_table.device)]


def couple(phi: torch.Tensor, tau_rows: torch.Tensor, beta: float) -> torch.Tensor:
    """Build ``Phi = [phi | beta * tau]`` for a batch (the joint coupling).

    Args:
        phi: ``(B, d_img)`` image features (carry grad).
        tau_rows: ``(B, d_txt)`` frozen per-row text embeddings (no grad).
        beta: text-block scale (:func:`text_weight_beta`).
    """
    text_block = (beta * tau_rows).to(phi.dtype)
    return torch.cat([phi, text_block], dim=1)


def couple_by_caption(phi: torch.Tensor, tau_table: torch.Tensor,
                      caption_ids: torch.Tensor, beta: float) -> torch.Tensor:
    """Convenience: gather ``tau(c)`` by caption id then concatenate onto ``phi``."""
    return couple(phi, gather_text(tau_table, caption_ids), beta)
