"""The iRDM training loop.

Each step draws a fresh large batch of ``N`` samples, generates them in one network
evaluation, embeds them under the battery of frozen encoders, scores each encoder with the
Nystrom MMD against its frozen full-data reference, self-normalizes and PID-weights the
per-encoder terms, sums, and takes one AdamW step -- with gradient caching absorbing the
memory of the batch-coupled loss. No teacher, adversary, or trajectory.

Distributed convention (matches the released code so the config learning rates are valid):
the fresh batch is sharded across ranks; the middle backward of the global-batch loss is
scaled by ``1/grad_accum`` and the per-parameter gradients are combined with an
``all_reduce`` **average**. The invariant ``N = batch * world * grad_accum`` (and
``num_chunks == grad_accum``) is enforced.
"""
from __future__ import annotations

import torch

from ..compare.grad_balance import clip_generator_grads, self_normalize
from ..compare.mmd_nystrom import mmd_nystrom_loss
from ..utils.distributed import all_reduce_mean, get_world_size
from .grad_caching import gradcache_backward


class Trainer:
    """Central loop wiring generator + battery + frozen references + AdamW + PID."""

    def __init__(self, generator, battery, references, optimizer, conditioner, *,
                 rollout_size: int, batch_size: int, grad_accum: int,
                 noise_shape, grad_clip: float = 2.0, gen_chunk: int = 8192,
                 pid=None, joint=None, device: str = "cuda"):
        world = get_world_size()
        if rollout_size != batch_size * world * grad_accum:
            raise ValueError(f"rollout_size {rollout_size} != batch*world*grad_accum "
                             f"({batch_size}*{world}*{grad_accum})")
        self.generator = generator
        self.battery = battery
        self.references = references          # {name: ReferencePack}
        self.optimizer = optimizer
        self.conditioner = conditioner
        self.batch_size = batch_size
        self.grad_accum = grad_accum          # == num_chunks per rank
        self.noise_shape = tuple(noise_shape)
        self.grad_clip = grad_clip
        self.gen_chunk = gen_chunk
        self.pid = pid
        self.joint = joint
        self.device = device
        self.step_idx = 0

    @property
    def _gen_params(self):
        return [p for p in self.generator.model.parameters() if p.requires_grad]

    def _encode(self, chunk):
        """encode_fn: (noise, condition) -> {name: (b, d)} (generate then embed + optional couple)."""
        noise, condition = chunk
        imgs = self.generator.sample(noise, self.conditioner.encode_for_generator(condition))
        feats = self.battery(imgs, only=set(self.references))
        if self.joint is not None:
            caption_ids = condition[1] if isinstance(condition, (tuple, list)) else None
            feats = self.joint.apply(feats, caption_ids)
        return feats

    def _loss(self, glob_feats):
        """loss_fn on the gathered global features: per-encoder Nystrom MMD, self-norm, PID-weight."""
        raws = {}
        for name, rp in self.references.items():
            raws[name] = mmd_nystrom_loss(glob_feats[name], rp.Z, rp.Z2, rp.alpha, rp.sigma,
                                          rp.k_rr, gen_chunk=self.gen_chunk)
        weights = self.pid.get_weights() if self.pid is not None else {}
        total = None
        for name, raw in raws.items():
            w = weights.get(name, self.references[name].weight)
            term = w * self_normalize(raw)
            total = term if total is None else total + term
        logs = {"raw_scores": {n: float(r.detach()) for n, r in raws.items()}}
        return total, logs

    def _make_chunks(self):
        """grad_accum fresh chunks of size batch_size (this rank's share of the rollout)."""
        chunks = []
        for _ in range(self.grad_accum):
            noise = torch.randn(self.batch_size, *self.noise_shape, device=self.device)
            cond = self.conditioner.sample(self.batch_size, device=self.device)
            chunks.append((noise, cond))
        return chunks

    def step(self) -> dict:
        """One optimizer step; returns logs (loss value + per-encoder raw scores)."""
        self.optimizer.zero_grad(set_to_none=True)
        chunks = self._make_chunks()
        loss_val, logs = gradcache_backward(
            chunks, self._encode, self._loss,
            scale=1.0 / self.grad_accum, gather=get_world_size() > 1)
        for p in self._gen_params:                         # combine ranks (AVG; config-lr convention)
            if p.grad is not None:
                all_reduce_mean(p.grad)
        grad_norm = clip_generator_grads(self._gen_params, self.grad_clip) if self.grad_clip > 0 else None
        if not torch.isfinite(torch.tensor(loss_val)):
            self.optimizer.zero_grad(set_to_none=True)     # skip a NaN step
        else:
            self.optimizer.step()
        if self.pid is not None:                           # update weights for the next step
            self.pid.step(logs["raw_scores"])
        self.step_idx += 1
        return {"step": self.step_idx, "loss": loss_val,
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                **logs}

    def train(self, num_steps: int, log_fn=None):
        """Run ``num_steps`` optimizer steps."""
        for _ in range(num_steps):
            logs = self.step()
            if log_fn is not None:
                log_fn(logs)
        return self.step_idx
