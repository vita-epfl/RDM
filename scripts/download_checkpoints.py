#!/usr/bin/env python
"""Fetch the released checkpoints + encoder weights.

The 14 panel encoders download lazily on first use (timm ``pretrained=True``, HF
``from_pretrained``, torch.hub) into ``HF_HOME`` / ``TORCH_HOME``; this script (a) pulls the
released pMF-H FD-SIM generator checkpoint and (b) optionally warms the encoder cache.
FLUX.2 klein-4B / AE weights and the external ``flux2`` package are fetched separately (see
docs/README.md). The FLUX VAE (held-out encoder) and SigLIP2 text tower download lazily too.

    python scripts/download_checkpoints.py --pmfh        # released pMF-H FD-SIM generator
    python scripts/download_checkpoints.py --warm-encoders [--eval]
"""
import argparse
import os

PMFH_REPO = "jjiaweiyang/FD-Loss"
PMFH_FILE = "checkpoints/post-trained/pMF-H_FD-SIM.pth"


def download_pmfh(out_dir="checkpoints"):
    """Fetch the generator weights and expose them at the path the configs' ``load_from`` uses.

    The HF file lives under a repo subdir; we download it (into the HF cache) and link the
    flat canonical name ``checkpoints/pMF-H_FD-SIM.pth`` that ``configs/imagenet.yaml``
    expects, so the path matches regardless of the repo's internal layout.
    """
    from huggingface_hub import hf_hub_download
    os.makedirs(out_dir, exist_ok=True)
    src = hf_hub_download(repo_id=PMFH_REPO, filename=PMFH_FILE)
    dst = os.path.join(out_dir, os.path.basename(PMFH_FILE))   # checkpoints/pMF-H_FD-SIM.pth
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:                                       # filesystems without symlinks
            import shutil
            shutil.copy2(src, dst)
    print("pMF-H FD-SIM ->", dst)
    return dst


def warm_encoders(eval_panel=False):
    from rdm.representation.checkpoints import warm_cache
    from rdm.representation.registry import all_specs, training_specs
    warm_cache(all_specs() if eval_panel else training_specs(), device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmfh", action="store_true", help="download the released pMF-H generator")
    ap.add_argument("--warm-encoders", action="store_true", help="prefetch encoder weights")
    ap.add_argument("--eval", action="store_true", help="include the 4 held-out encoders")
    args = ap.parse_args()
    if args.pmfh:
        download_pmfh()
    if args.warm_encoders:
        warm_encoders(eval_panel=args.eval)
    if not (args.pmfh or args.warm_encoders):
        ap.print_help()


if __name__ == "__main__":
    main()
