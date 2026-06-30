"""Distributed entrypoint: load config, build everything, run the loop, checkpoint.

``python -m rdm.train.launch configs/imagenet.yaml`` (under ``torchrun`` for multi-GPU).
Config is YAML with an ``extends:`` chain (parent resolved relative to the child). The
generator is the only trainable module (encoders frozen, replicated); gradients are combined
manually via the GradCache all-gather + an all-reduce average, so there is no DDP wrapper.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

logger = logging.getLogger("rdm")

from ..representation.battery import Battery
from ..representation.generators import build_generator
from ..representation.models import convert_pmf_checkpoint, pMFDenoiser_models
from ..utils.distributed import is_main_process, setup_distributed
from ..utils.io import save_bundle
from ..utils.distributed import get_world_size
from ..utils.logging import setup_logging, setup_wandb
from ..utils.seed import fix_random_seeds
from .data import ClassConditioner, PromptConditioner
from .joint_objective import JointObjective
from .pid_lagrangian import build_pid_lagrangian
from .references import load_floors, load_reference_packs, load_text_table
from .trainer import Trainer


def load_config(path: str) -> SimpleNamespace:
    """Recursive YAML loader with an ``extends:`` chain (child keys override parent)."""
    path = Path(path).resolve()
    raw = yaml.safe_load(open(path)) or {}
    merged: dict = {}
    if "extends" in raw:
        parent = load_config((path.parent / raw["extends"]).resolve())
        merged.update(vars(parent))
        del raw["extends"]
    merged.update(raw)
    return SimpleNamespace(**merged)


def build_adamw(params, cfg) -> torch.optim.Optimizer:
    """AdamW with weight decay 0 (the iRDM optimizer)."""
    return torch.optim.AdamW(params, lr=cfg.lr, betas=(getattr(cfg, "beta1", 0.9),
                             getattr(cfg, "beta2", 0.95)), weight_decay=getattr(cfg, "weight_decay", 0.0))


def load_generator_weights(model, load_from: str, mode: str) -> None:
    """Load a generator checkpoint into ``model`` for BOTH modes.

    ``load_from`` is either the warm-start base (training) or the post-trained student
    (evaluation). The checkpoint is ``{"model": state_dict, ...}`` as written by
    :func:`save_checkpoint` (or a bare state_dict). The ImageNet pMF path remaps keys via
    :func:`convert_pmf_checkpoint`; the FLUX path loads the ``Flux2AdapterModel`` state_dict
    directly (same module that was saved). Without this, FLUX eval would silently run the
    untrained klein-4B base instead of the student.
    """
    if not load_from:
        return
    if not os.path.exists(load_from):
        # Documented contract: a missing checkpoint runs the untrained base, it does not crash
        # (so eval-flux won't die only after encoding all the prompts on the GPU).
        logger.warning("[load_from] %s not found -- skipping; running the UNTRAINED base. "
                       "Point load_from at a trained checkpoint to evaluate the student.", load_from)
        return
    sd = torch.load(load_from, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    if mode != "flux":
        sd = convert_pmf_checkpoint(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        logger.warning("[load_from] %s: %d missing / %d unexpected keys (missing e.g. %s; "
                       "unexpected e.g. %s)", load_from, len(missing), len(unexpected),
                       list(missing)[:3], list(unexpected)[:3])


def build_generator_from_config(cfg, device: str = "cuda"):
    """Construct the trainable generator (+ optional tokenizer) and its sampling wrapper."""
    mode = getattr(cfg, "mode", "imagenet")
    sampling_args = dict(num_steps=getattr(cfg, "num_sampling_steps", 1),
                         t_min=getattr(cfg, "interval_min", 0.4),
                         t_max=getattr(cfg, "interval_max", 0.65),
                         cfg=getattr(cfg, "cfg", 1.0))
    tokenizer = None
    if mode == "flux":
        from ..representation.generators.flux_generator import Flux2AdapterModel, Flux2VAETokenizer
        model = Flux2AdapterModel(image_resolution=getattr(cfg, "img_size", 512),
                                  param_dtype=torch.float32,
                                  gradient_checkpointing=getattr(cfg, "grad_checkpointing", True),
                                  compile_blocks=getattr(cfg, "compile_model", False)).to(device)
        tokenizer = Flux2VAETokenizer(device=device)
    else:
        model = pMFDenoiser_models[cfg.model](
            img_size=getattr(cfg, "img_size", 256), num_classes=getattr(cfg, "num_classes", 1000),
            noise_scale=getattr(cfg, "noise_scale", 2.0), rope_2d=getattr(cfg, "rope_2d", True),
            learned_pe=getattr(cfg, "learned_pe", True),
            disable_v_head=getattr(cfg, "disable_v_head", True)).to(device)
    load_generator_weights(model, getattr(cfg, "load_from", ""), mode)
    generator = build_generator(mode, model, sampling_args, args=cfg, tokenizer=tokenizer)
    return generator, model


def build_trainer(cfg, device: str = "cuda") -> Trainer:
    """Wire generator + battery + references + conditioner + PID + joint into a Trainer."""
    generator, model = build_generator_from_config(cfg, device)
    names = list(cfg.encoders)                                   # training encoder short names
    bundle_paths = dict(zip(names, cfg.nystrom_paths))
    weights = dict(zip(names, getattr(cfg, "encoder_weights", [1.0] * len(names))))
    references = load_reference_packs(bundle_paths, weights, device=device)
    battery = Battery([__import__("rdm.representation.registry", fromlist=["by_name"]).by_name(n)
                       for n in names], device=device)
    if getattr(cfg, "battery_bf16", False):                  # fit smaller GPUs: store the frozen ViT
        for enc in battery.encoders.values():                # encoder WEIGHTS in bf16 (~10 GB saved).
            if not getattr(enc, "has_logits", False):        # Their forward already runs under bf16
                enc.to(torch.bfloat16)                       # autocast, so features are ~unchanged;
        logger.info("[battery] frozen ViT encoders cast to bf16 (Inception kept fp32)")  # Inception fp32.

    pid = None
    if getattr(cfg, "pid_enable", False):
        floors = load_floors(cfg.pid_floors_json)
        pid = build_pid_lagrangian(cfg, names, floors)

    joint = None
    if getattr(cfg, "joint_enable", False):
        tau = load_text_table(cfg.joint_text_psi, device=device)
        # text-block scale beta = sigma_img / s_txt, frozen into each joint bundle by
        # build_joint_one (fall back to sigma_img if a bundle predates the stored beta).
        betas = {n: (references[n].beta if references[n].beta is not None else references[n].sigma)
                 for n in names}
        joint = JointObjective(tau, betas, enabled=True)

    if getattr(cfg, "mode", "imagenet") == "flux":
        noise_shape = (model.in_channels, model.input_size, model.input_size)
    else:
        noise_shape = (3, getattr(cfg, "img_size", 256), getattr(cfg, "img_size", 256))

    optimizer = build_adamw([p for p in model.parameters() if p.requires_grad], cfg)
    return Trainer(generator, battery, references, optimizer,
                   ClassConditioner(getattr(cfg, "num_classes", 1000)) if joint is None
                   else PromptConditioner(load_text_table(cfg.ctx_pool, device="cpu")),
                   rollout_size=cfg.rollout_size, batch_size=cfg.batch_size,
                   grad_accum=cfg.grad_accum, noise_shape=noise_shape,
                   grad_clip=getattr(cfg, "grad_clip", 2.0), gen_chunk=getattr(cfg, "gen_chunk", 8192),
                   pid=pid, joint=joint, device=device)


def save_checkpoint(trainer: Trainer, out_dir: str) -> None:
    if is_main_process():
        path = os.path.join(out_dir, f"step_{trainer.step_idx:07d}.pth")
        save_bundle({"model": trainer.generator.model.state_dict(), "step": trainer.step_idx}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rank, world, local_rank = setup_distributed()
    setup_logging(rank=rank)
    fix_random_seeds(getattr(cfg, "seed", 0) + rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    trainer = build_trainer(cfg, device=device)
    out_dir = os.path.join(getattr(cfg, "output_dir", "./work_dirs"), getattr(cfg, "exp_name", "run"))
    os.makedirs(out_dir, exist_ok=True)

    # wandb (main process only); metric schema mirrors the original FD-Loss run (train/ + perf/).
    wandb_run = None
    if is_main_process():
        wandb_run = setup_wandb(project=getattr(cfg, "wandb_project", "irdm"),
                                name=getattr(cfg, "wandb_name", None) or getattr(cfg, "exp_name", "run"),
                                config=vars(cfg), enabled=getattr(cfg, "wandb_enable", False),
                                entity=getattr(cfg, "wandb_entity", None), resume="allow")

    total = int(cfg.steps)
    print_freq = getattr(cfg, "print_freq", 1)
    session_start = time.time()
    samples_seen = 0
    for _ in range(total):
        t0 = time.time()
        logs = trainer.step()
        dt = max(time.time() - t0, 1e-9)
        samples_seen += cfg.rollout_size
        if is_main_process() and trainer.step_idx % print_freq == 0:
            print(f"step {logs['step']}  loss {logs['loss']:.5f}  grad_norm {logs['grad_norm']}")
        if wandb_run is not None and trainer.step_idx % print_freq == 0:
            step = logs["step"]
            sps = cfg.rollout_size / dt                                       # global throughput
            mem_gb = (torch.cuda.max_memory_reserved() / 1e9) if torch.cuda.is_available() else 0.0
            elapsed = time.time() - session_start
            eta = elapsed / step * (total - step) if step > 0 else 0.0
            metrics = {"train/loss": logs["loss"], "train/grad_norm": logs["grad_norm"],
                       "train/lr": trainer.optimizer.param_groups[0]["lr"],
                       "train/samples_seen_M": samples_seen / 1e6,
                       "perf/samples_per_sec": sps, "perf/samples_per_sec_per_device": sps / world,
                       "perf/max_reserved_mem_gb": mem_gb,
                       "perf/elapsed_real_hours": elapsed / 3600,
                       "perf/elapsed_device_hours": elapsed / 3600 * world,
                       "perf/eta_real_hours": eta / 3600,
                       **{f"train/{k}": v for k, v in logs.get("raw_scores", {}).items()}}
            wandb_run.log({k: v for k, v in metrics.items() if v is not None}, step=step)
        if trainer.step_idx % getattr(cfg, "save_freq", 100) == 0:
            save_checkpoint(trainer, out_dir)
    save_checkpoint(trainer, out_dir)
    if is_main_process() and torch.cuda.is_available():
        print(f"[mem] peak reserved {torch.cuda.max_memory_reserved() / 1e9:.1f} GB "
              f"of {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
