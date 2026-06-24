"""Distributed entrypoint: load config, build everything, run the loop, checkpoint.

``python -m rdm.train.launch configs/imagenet.yaml`` (under ``torchrun`` for multi-GPU).
Config is YAML with an ``extends:`` chain (parent resolved relative to the child). The
generator is the only trainable module (encoders frozen, replicated); gradients are combined
manually via the GradCache all-gather + an all-reduce average, so there is no DDP wrapper.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from ..representation.battery import Battery
from ..representation.generators import build_generator
from ..representation.models import convert_pmf_checkpoint, pMFDenoiser_models
from ..utils.distributed import is_main_process, setup_distributed
from ..utils.io import save_bundle
from ..utils.logging import setup_logging
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
        if getattr(cfg, "load_from", ""):
            sd = torch.load(cfg.load_from, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd)
            model.load_state_dict(convert_pmf_checkpoint(sd), strict=False)
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
    total = int(cfg.steps)
    for _ in range(total):
        logs = trainer.step()
        if is_main_process() and trainer.step_idx % getattr(cfg, "print_freq", 1) == 0:
            print(f"step {logs['step']}  loss {logs['loss']:.5f}  grad_norm {logs['grad_norm']}")
        if trainer.step_idx % getattr(cfg, "save_freq", 100) == 0:
            save_checkpoint(trainer, out_dir)
    save_checkpoint(trainer, out_dir)


if __name__ == "__main__":
    main()
