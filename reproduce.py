#!/usr/bin/env python
"""One command per paper artifact.

    python reproduce.py fig3 [--smoke]        # Fig. 3 spiral grid (self-contained)
    python reproduce.py fig4 [--smoke]        # Fig. 4 batch-size axis (low-dim)
    python reproduce.py ablation-distance     # Table 4 distance ordering (low-dim spiral)
    python reproduce.py train-imagenet        # post-train pMF-H  (torchrun via scripts/train.sh)
    python reproduce.py train-flux            # post-train FLUX.2 klein (joint)
    python reproduce.py eval-imagenet         # SW_r14 + MMDr14 + PickScore
    python reproduce.py eval-flux             # GenEval + PickScore

See docs/reproduction_map.md for the artifact -> command -> paper-table map. Full-scale
artifacts (train/eval) need the downloaded weights + precomputed reference bundles
(scripts/download_checkpoints.py, scripts/run_refprep.sh); the toy artifacts are
self-contained.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default=None)
    args, rest = ap.parse_known_args()
    n_iters = 120 if args.smoke else 8000

    if args.artifact == "fig3":
        from rdm.figures import fig3
        print("saved", fig3(n_iters=n_iters))
    elif args.artifact == "fig4":
        from rdm.figures import fig4
        print("saved", fig4(n_iters=n_iters))
    elif args.artifact == "ablation-distance":
        from rdm.toy.run_ablations import run
        out = run(n_iters=n_iters)
        print("ranking (best->worst medDist):", " > ".join(out["ranking"]))
    elif args.artifact in ("train-imagenet", "train-flux"):
        # Training is multi-GPU (torchrun); the rollout = batch*world*grad_accum invariant
        # assumes the configured world size. Delegate to the launcher (honors $GPUS).
        import subprocess
        config = args.config or f"configs/{args.artifact.split('-')[1]}.yaml"
        print(f"launching: GPUS=${{GPUS:-8}} bash scripts/train.sh {config}")
        subprocess.run(["bash", "scripts/train.sh", config], check=True)
    elif args.artifact == "eval-flux":
        import torch

        from rdm.data.prompts import load_geneval_metadata, load_jsonl_prompts
        from rdm.eval.flux_eval import evaluate_flux
        from rdm.representation.flux_text_context import Flux2TextContextEncoder
        from rdm.train.launch import build_generator_from_config, load_config
        cfg = load_config(args.config or "configs/eval_flux.yaml")
        device = "cuda"
        geneval_metadata = load_geneval_metadata(cfg.geneval_metadata)
        geneval_prompts = [m["prompt"] for m in geneval_metadata]
        pickscore_prompts = load_jsonl_prompts(cfg.pickscore_prompts)
        # The FLUX context fed to the generator must be the Qwen3 encoding of the ACTUAL eval
        # prompts -- NOT the COCO ctx_pool (slicing that renders COCO captions and scores them
        # against the eval prompts). ctx_len must match the training ctx_pool's sequence length.
        ctx_len = getattr(cfg, "flux_ctx_len", None)
        if ctx_len is None and getattr(cfg, "ctx_pool", None) and os.path.exists(cfg.ctx_pool):
            from rdm.train.references import load_text_table
            ctx_len = int(load_text_table(cfg.ctx_pool).shape[1])
        ctx_len = int(ctx_len or 48)
        enc = Flux2TextContextEncoder(ctx_len=ctx_len, model_id=getattr(cfg, "flux_text_model", None),
                                      flux2_src=getattr(cfg, "flux2_src", None), device=device)
        geneval_ctx = enc.encode(geneval_prompts)
        pickscore_ctx = enc.encode(pickscore_prompts)
        del enc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gen, _ = build_generator_from_config(cfg, device=device)   # loads cfg.load_from (student)
        res = evaluate_flux(
            gen,
            pickscore_prompts=pickscore_prompts, pickscore_ctx=pickscore_ctx,
            geneval_metadata=geneval_metadata, geneval_ctx=geneval_ctx,
            n_geneval_per_prompt=getattr(cfg, "geneval_n_per_prompt", 4),
            geneval_repo=getattr(cfg, "geneval_repo", None), device=device)
        print("FLUX eval:", res)
    elif args.artifact == "eval-imagenet":
        from rdm.eval.imagenet_eval import evaluate_imagenet, load_eval_banks
        from rdm.eval.report import format_table
        from rdm.train.launch import build_generator_from_config, load_config
        cfg = load_config(args.config or "configs/eval_imagenet.yaml")
        gen, _ = build_generator_from_config(cfg)
        train_feats, val_feats, rff_banks, floors = load_eval_banks(
            cfg.eval_rff_bundle_dir, cfg.sw_bank_dir, cfg.mmdr14_floors_json)
        # Fig. 6 PickScore prompts: a custom JSON list if configured, else the torchvision
        # ImageNet class names (offline, no asset) -- so PickScore runs out of the box.
        cp_path = getattr(cfg, "imagenet_class_prompts", None)
        if cp_path and os.path.exists(cp_path):
            from rdm.utils.io import read_json
            class_prompts = read_json(cp_path)
        else:
            from rdm.data.prompts import imagenet_class_prompts_torchvision
            class_prompts = imagenet_class_prompts_torchvision()
        res = evaluate_imagenet(
            gen, train_feats=train_feats, val_feats=val_feats, rff_banks=rff_banks,
            mmd_val_floors=floors, n_samples=getattr(cfg, "sw_n_samples", 16384),
            num_classes=getattr(cfg, "num_classes", 1000), img_size=getattr(cfg, "img_size", 256),
            class_prompts=class_prompts, pickscore_n=getattr(cfg, "pickscore_n_latents", 4000))
        ps = f"   PickScore = {res['pickscore']:.3f}" if "pickscore" in res else ""
        print(f"SW_r14 = {res['sw']['swr14']:.3f}   MMDr14 = {res['mmd']['mmdr14']:.3f}{ps}")
        print(format_table([res["reports"][0]], "SW_r14"))
        print(format_table([res["reports"][1]], "MMDr14"))
    else:
        print(f"unknown artifact {args.artifact!r}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
