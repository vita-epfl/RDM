#!/usr/bin/env python
"""Build the FLUX.2 Qwen3 text-context table (.npy) the FLUX generator conditions on.

This is the in-repo builder for the one FLUX artifact that previously had none: the
generator's text conditioning. The SAME encoder produces (a) the training ``ctx_pool``
(``data/fid_stats/flux2/qwen3_ctx_coco.npy``) the loop samples over the COCO captions, and
(b) any eval prompt set. It runs the external FLUX.2 Qwen3 text encoder
(:class:`rdm.representation.flux_text_context.Flux2TextContextEncoder`) over a prompt source
and writes a ``(N, ctx_len, 7680)`` float16 table via a memmap, so the 82K-caption COCO pool
never lands in RAM.

    # training ctx_pool over the COCO captions (one row per caption, row-aligned to coco_pairs):
    python scripts/build_flux2_ctx.py --captions data/coco/coco_pairs.npz \
        --out data/fid_stats/flux2/qwen3_ctx_coco.npy --ctx-len 48
    # any prompt set (.jsonl with a "prompt" key) -- e.g. precompute the eval contexts:
    python scripts/build_flux2_ctx.py --jsonl assets/pickapic_test_prompts.jsonl \
        --out data/fid_stats/flux2/qwen3_ctx_pickapic.npy --ctx-len 48

Needs the external ``flux2`` package (``FLUX2_SRC`` env or ``--flux2-src``) + a GPU. ``ctx_len``
MUST match between the training ctx_pool and ``flux_ctx_len`` in the eval config. See
docs/flux_reference.md.
"""
import argparse
import json
import os

import numpy as np
import torch

from rdm.representation.flux_text_context import FLUX2_QWEN3_DIM, Flux2TextContextEncoder


def _load_prompts(args) -> list[str]:
    if args.captions:
        from rdm.data.coco import load_coco_pairs
        return [str(c) for c in load_coco_pairs(args.captions)["captions"]]
    if args.jsonl:
        out = []
        with open(args.jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(str(json.loads(line)[args.key]))
        return out
    raise SystemExit("need --captions <coco_pairs.npz> or --jsonl <prompts.jsonl>")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--captions", help="COCO pairing .npz (rows = captions, row-aligned)")
    src.add_argument("--jsonl", help="prompt .jsonl (one JSON object per line)")
    ap.add_argument("--key", default="prompt", help="jsonl field holding the prompt string")
    ap.add_argument("--out", required=True, help="output .npy")
    ap.add_argument("--ctx-len", type=int, default=48, help="sequence length L (match training!)")
    ap.add_argument("--variant", default="4B")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--flux2-src", default=None, help="path to the flux2 package (else FLUX2_SRC)")
    args = ap.parse_args()

    prompts = _load_prompts(args)
    print(f"[flux2-ctx] {len(prompts)} prompts -> {args.out} "
          f"(ctx_len={args.ctx_len}, dim={FLUX2_QWEN3_DIM})")
    enc = Flux2TextContextEncoder(args.ctx_len, variant=args.variant,
                                  flux2_src=args.flux2_src, device="cuda")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mm = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float16,
                                   shape=(len(prompts), args.ctx_len, FLUX2_QWEN3_DIM))
    for lo in range(0, len(prompts), args.batch):
        ctx = enc.embedder([str(p) for p in prompts[lo:lo + args.batch]])  # (b, ctx_len, 7680) bf16
        mm[lo:lo + ctx.shape[0]] = ctx.to(torch.float16).cpu().numpy()
        if lo % (args.batch * 50) == 0:
            print(f"  {lo}/{len(prompts)}")
    mm.flush()
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump({"n": len(prompts), "ctx_len": args.ctx_len, "dim": FLUX2_QWEN3_DIM,
                   "variant": args.variant}, f, indent=2)
    print(f"[flux2-ctx] wrote {args.out} shape=({len(prompts)}, {args.ctx_len}, {FLUX2_QWEN3_DIM})")


if __name__ == "__main__":
    main()
