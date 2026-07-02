#!/usr/bin/env python
"""Fetch the released checkpoints + (optionally) warm the encoder cache.

Two released generators back the headline evals; both are plain training checkpoints (a dict
with the weights under the ``model`` key) that drop straight into the configs' ``load_from``:

    --pmfh   ImageNet-256 pMF-H FD-SIM one-step generator   (SW_r14, MMDr14, PickScore)
             https://huggingface.co/Lanl11/pMF-H-FDSIM-imagenet256-sigma07-4k   (model.pth, 3.8 GB)
    --flux   FLUX.2 klein-4B one-step student (geALLcoco s180) (GenEval, PickScore)
             https://huggingface.co/epfl-vita/flux2-klein-1step-rdm   (…_s180.pth, 15.5 GB fp32)

Each file is pulled into the HF cache and linked to the flat canonical name the eval configs
reference, so the download -> eval flow works out of the box:

    checkpoints/pMF-H_FD-SIM.pth                      <- configs/eval_imagenet.yaml load_from
    checkpoints/flux2_klein_1step_rdm_geallcoco_s180.pth  <- configs/eval_flux.yaml load_from

The panel encoders (and, for FLUX, the klein VAE + the Qwen3 text encoder that encodes prompts
-- ``Qwen/Qwen3-4B-FP8``, or ``Qwen/Qwen3-4B`` bf16 on offline nodes) download lazily on first
use into ``HF_HOME`` / ``TORCH_HOME``. ``--warm-encoders`` prefetches the 10 training encoders;
add ``--eval`` to also warm the 4 held-out encoders (all 14).

    python scripts/download_checkpoints.py --pmfh        # ImageNet generator
    python scripts/download_checkpoints.py --flux        # FLUX one-step student
    python scripts/download_checkpoints.py --warm-encoders [--eval]

Both checkpoint repos are public. Repo overrides (if you mirror or re-host): ``--pmfh-repo`` /
``--flux-repo`` (or the ``PMFH_REPO`` / ``FLUX_REPO`` env vars). The download still degrades
gracefully with a clear message if a repo is renamed, gated, or unreachable.
"""
import argparse
import os

# repo_id, filename-in-repo, canonical flat name under checkpoints/ (== the configs' load_from)
PMFH = (os.environ.get("PMFH_REPO", "Lanl11/pMF-H-FDSIM-imagenet256-sigma07-4k"),
        "model.pth", "pMF-H_FD-SIM.pth")
FLUX = (os.environ.get("FLUX_REPO", "epfl-vita/flux2-klein-1step-rdm"),
        "flux2_klein_1step_rdm_geallcoco_s180.pth", "flux2_klein_1step_rdm_geallcoco_s180.pth")


def download_and_link(repo_id, filename, dst_name, out_dir="checkpoints"):
    """Fetch ``repo_id/filename`` into the HF cache and link ``checkpoints/<dst_name>`` to it.

    Linking the canonical flat name keeps the configs' ``load_from`` valid regardless of the
    repo's internal layout; falls back to a copy on filesystems without symlinks.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    os.makedirs(out_dir, exist_ok=True)
    try:
        src = hf_hub_download(repo_id=repo_id, filename=filename)
    except (RepositoryNotFoundError, GatedRepoError) as e:
        raise SystemExit(
            f"[download] cannot access {repo_id!r} ({type(e).__name__}). If it was renamed or gated, "
            f"`huggingface-cli login` with an authorized account, or override the repo with the "
            f"matching --*-repo flag / env var.") from e
    dst = os.path.join(out_dir, dst_name)
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(os.path.abspath(src), dst)
        except OSError:                                       # filesystems without symlinks
            import shutil
            shutil.copy2(src, dst)
    print(f"{repo_id} -> {dst}")
    return dst


def warm_encoders(eval_panel=False):
    from rdm.representation.checkpoints import warm_cache
    from rdm.representation.registry import all_specs, training_specs
    warm_cache(all_specs() if eval_panel else training_specs(), device="cpu")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmfh", action="store_true", help="download the released ImageNet pMF-H generator")
    ap.add_argument("--flux", action="store_true", help="download the released FLUX.2 one-step student")
    ap.add_argument("--pmfh-repo", default=PMFH[0], help="override the ImageNet checkpoint HF repo")
    ap.add_argument("--flux-repo", default=FLUX[0], help="override the FLUX checkpoint HF repo")
    ap.add_argument("--warm-encoders", action="store_true", help="prefetch encoder weights")
    ap.add_argument("--eval", action="store_true", help="include the 4 held-out encoders when warming")
    args = ap.parse_args()
    if args.pmfh:
        download_and_link(args.pmfh_repo, PMFH[1], PMFH[2])
    if args.flux:
        download_and_link(args.flux_repo, FLUX[1], FLUX[2])
    if args.warm_encoders:
        warm_encoders(eval_panel=args.eval)
    if not (args.pmfh or args.flux or args.warm_encoders):
        ap.print_help()


if __name__ == "__main__":
    main()
