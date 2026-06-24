#!/usr/bin/env python
"""Fail fast if the reference artifacts a run needs are missing.

Validates the per-encoder Nystrom bundles, the PID floors, and (for eval) the RFF / SW
banks referenced by a config, and reports the on-disk footprint -- run before a multi-node
job so a missing bundle surfaces immediately rather than mid-run.

    python scripts/check_artifacts.py configs/imagenet.yaml
"""
import argparse
import os
import sys


def _exists(path):
    return path and os.path.exists(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    from rdm.train.launch import load_config
    cfg = load_config(args.config)

    missing, total_bytes = [], 0
    # Single-file artifacts (training + eval): training bundles, the warm-start checkpoint,
    # PID / MMDr14 floor tables, the joint text table, the FLUX context pool, and the
    # external eval prompt assets.
    paths = list(getattr(cfg, "nystrom_paths", []) or [])
    for k in ("load_from", "pid_floors_json", "joint_text_psi", "ctx_pool",
              "mmdr14_floors_json", "geneval_metadata", "pickscore_prompts"):
        v = getattr(cfg, k, None)
        if v:
            paths.append(v)
    for p in paths:
        if _exists(p):
            total_bytes += os.path.getsize(p) if os.path.isfile(p) else 0
        else:
            missing.append(p)

    # Per-encoder eval bank directories (RFF / SW): present + non-empty.
    for k in ("eval_rff_bundle_dir", "sw_bank_dir"):
        d = getattr(cfg, k, None)
        if not d:
            continue
        pts = [f for f in os.listdir(d) if f.endswith(".pt")] if os.path.isdir(d) else []
        if not pts:
            missing.append(f"{d}/ ({k}: directory missing or contains no *.pt banks)")
        else:
            total_bytes += sum(os.path.getsize(os.path.join(d, f)) for f in pts)
            print(f"  {k}: {len(pts)} bank file(s) in {d}")

    print(f"checked {len(paths)} file artifacts; present footprint {total_bytes / 1e9:.2f} GB")
    if missing:
        print("MISSING:", file=sys.stderr)
        for m in missing:
            print("  ", m, file=sys.stderr)
        sys.exit(1)
    print("all artifacts present.")


if __name__ == "__main__":
    main()
