"""GradCache (Gao et al. 2021) for the batch-coupled iRDM loss.

The within-batch repulsion averages the kernel over all NxN pairs, so the loss does NOT
decompose into a sum of per-sample terms and plain gradient accumulation is invalid;
holding the full autograd graph for N generated samples + encoder forwards OOMs. GradCache
computes the exact full-batch gradient at one-micro-batch memory:

  Pass 1 (no grad): generate + encode all chunks -> cached detached features F (graphs freed).
  Middle:           one backward of L(F) w.r.t. F (tiny kernel graph) -> cached G = dL/dF.
  Pass 2 (grad):    re-encode each chunk one at a time and backprop the cached G slice via
                    the surrogate ``sum(feat * G_slice.detach())`` -> exact dL/dtheta.

The accumulated gradient is elementwise-identical to a naive full-batch backward under
deterministic fp32 with equal pass-1/pass-2 chunk sizes (proved in ``tests``), and a
first-order approximation under bf16 autocast with unequal chunk sizes (a valid descent
direction). Multi-GPU: the middle backward routes the gathered global ``dL/dF`` back to each
rank's local rows; an outer ``all_reduce`` mean (in the trainer) combines ranks.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch

from ..utils.distributed import diff_all_gather


def gradcache_backward(chunk_inputs: List, encode_fn: Callable[[object], Dict[str, torch.Tensor]],
                       loss_fn: Callable[[Dict[str, torch.Tensor]], Tuple[torch.Tensor, dict]],
                       *, scale: float = 1.0, gather: bool = True) -> Tuple[float, dict]:
    """Full-batch GradCache step; ACCUMULATES dL/dtheta into the parameters (no optimizer step).

    Args:
        chunk_inputs: per-micro-batch inputs for this rank (iterated in the same order in
            both passes; ``encode_fn`` must be deterministic in its input).
        encode_fn: ``input -> {encoder_name: (b, d)}`` (builds generator+encoder graph).
        loss_fn: ``{encoder_name: (N_global, d) leaf} -> (scalar_loss, logs)`` (the full-batch
            per-encoder MMD-Nystrom + self-norm + PID weights on the gathered global features).
        scale: multiply the loss before backward (1.0 for a single full-batch step).
        gather: all-gather local features to the global batch before ``loss_fn`` (False in tests).
    """
    if not chunk_inputs:
        raise ValueError("gradcache_backward: chunk_inputs is empty")

    # ---- Pass 1: no-grad features, freed graphs ----
    reps: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for ci in chunk_inputs:
            for k, v in encode_fn(ci).items():
                reps.setdefault(k, []).append(v.detach())
    leaves = {k: torch.cat(v, 0).detach().requires_grad_(True) for k, v in reps.items()}
    chunk_sizes = [next(iter(reps.values()))[i].shape[0] for i in range(len(chunk_inputs))]
    for k, v in reps.items():
        if [t.shape[0] for t in v] != chunk_sizes:
            raise RuntimeError(f"gradcache_backward: encoder '{k}' chunk sizes differ")

    # ---- Middle: full-batch loss backward w.r.t. cached features ----
    glob = {k: (diff_all_gather(v) if gather else v) for k, v in leaves.items()}
    loss, logs = loss_fn(glob)
    (loss * scale).backward()
    G_local = {k: leaves[k].grad.detach() for k in leaves}
    loss_val = float(loss.detach())

    # ---- Pass 2: re-forward per chunk, surrogate backward into theta ----
    offsets = {k: 0 for k in leaves}
    for ci in chunk_inputs:
        feats = encode_fn(ci)
        surrogate = None
        for k, fm in feats.items():
            b = fm.shape[0]
            g = G_local[k][offsets[k]:offsets[k] + b]
            offsets[k] += b
            term = (fm * g.detach()).sum()
            surrogate = term if surrogate is None else surrogate + term
        surrogate.backward()

    return loss_val, logs
