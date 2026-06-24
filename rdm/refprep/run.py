"""End-to-end driver for the offline reference precompute.

Turns the :mod:`rdm.refprep` library functions into one runnable pipeline so the README's
"build the frozen reference" step is a single command rather than a manual loop. Two modes:

* ``imagenet`` -- the ImageNet-256 reference everything in :mod:`rdm.train` /
  :mod:`rdm.eval` consumes. For each encoder it streams the train / val images through the
  frozen backbone (the heavy compute), then builds, with output paths matching the configs:

  - ``bundles/nystrom/<enc>_nystrom_M4096.pt``  -- the training Nystrom reference (10 train enc);
  - ``imagenet_val_floors.json``                -- the PID real-validation floors (10 train enc);
  - ``bundles/eval_rff/<enc>.pt`` + ``mmdr14_val_floors.json`` -- the MMDr14 eval banks (14 enc);
  - ``bundles/eval_sw/<enc>_{train,val}.pt``     -- the SW_r14 real-feature banks (14 enc).

* ``joint`` -- the FLUX image-text joint reference: the frozen SigLIP2 text table tau(c) and
  the per-encoder joint Nystrom bundles over ``[phi(x) | beta*tau(c)]``
  (``bundles/nystrom_joint/<enc>_joint_M4096.pt``).

One process loops the chosen encoders sequentially. To shard the heavy extraction across
GPUs, launch several invocations with disjoint ``--encoders`` on different ``CUDA_VISIBLE_DEVICES``;
the per-encoder artifacts are independent and the combined floor JSONs are rebuilt from
per-encoder parts on every run, so the last shard to finish writes the complete table.

    # ImageNet (all 14 encoders, single GPU):
    python -m rdm.refprep.run imagenet --train <imagenet/train> --val <imagenet/val>
    # shard example (encoder subset on GPU 2):
    CUDA_VISIBLE_DEVICES=2 python -m rdm.refprep.run imagenet --train ... --val ... \
        --encoders dinov3_l,siglip2
    # FLUX joint reference (needs the COCO pairing from scripts/prepare_datasets.py):
    python -m rdm.refprep.run joint --coco-pairs data/coco/coco_pairs.npz
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

import torch

from ..representation.registry import all_specs, by_name, training_specs
from ..utils.io import read_json, write_json
from .build_eval_reference import build_rff_bank, build_sw_bank
from .build_nystrom_reference import build_one
from .build_joint_reference import build_joint_one
from .compute_floors import compute_floor
from .extract_features import cache_median_sigma, extract_features, save_features

logger = logging.getLogger("rdm")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".JPEG"}

# RFF mean / SW banks / floors are estimated on a real subsample, not the full train set
# (rff_phi over 1.28M x 4096 would OOM, and the metric only needs a representative sample).
DEFAULT_EVAL_SUBSAMPLE = 50000


# --------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------
def _collect_images(root: str) -> list[str]:
    """All image paths under ``root`` (recursive, sorted) -- handles flat or class-nested roots."""
    paths = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if os.path.splitext(p)[1] in IMAGE_EXTS]
    if not paths:
        raise FileNotFoundError(f"no images under {root}")
    return sorted(paths)


def _image_dataset(root: str, img_size: int, max_images: int | None):
    from ..data.imagenet import ImageListDataset
    paths = _collect_images(root)
    if max_images:
        paths = paths[:max_images]
    logger.info("[refprep] %s -> %d images", root, len(paths))
    return ImageListDataset(paths, img_size=img_size)


# --------------------------------------------------------------------------------------
# combined-floor JSONs rebuilt from per-encoder parts (shard-safe)
# --------------------------------------------------------------------------------------
def _write_floor_part(out: str, kind: str, name: str, value: float) -> None:
    write_json({name: float(value)}, os.path.join(out, "floors_parts", kind, f"{name}.json"))


def _rebuild_floor_json(out: str, kind: str, dest: str) -> None:
    """Merge every per-encoder floor part of ``kind`` into one ``{name: value}`` JSON."""
    parts = glob.glob(os.path.join(out, "floors_parts", kind, "*.json"))
    merged: dict = {}
    for p in sorted(parts):
        merged.update(read_json(p))
    if merged:
        write_json(merged, os.path.join(out, dest))
        logger.info("[refprep] wrote %s (%d encoders)", dest, len(merged))


# --------------------------------------------------------------------------------------
# imagenet reference
# --------------------------------------------------------------------------------------
def run_imagenet(train_root: str, val_root: str, out: str, encoders: list[str], *,
                 device: str = "cuda", img_size: int = 256, extract_bs: int = 256,
                 num_workers: int = 8, n_landmarks: int = 4096, eval_subsample: int = DEFAULT_EVAL_SUBSAMPLE,
                 max_images: int | None = None, max_ref_samples: int | None = None,
                 skip_existing: bool = False) -> None:
    """Build every ImageNet reference artifact for ``encoders`` (train + eval banks)."""
    train_ds = _image_dataset(train_root, img_size, max_images)
    val_ds = _image_dataset(val_root, img_size, max_images)
    train_names = {s.name for s in training_specs()}

    for name in encoders:
        spec = by_name(name)
        nys_path = os.path.join(out, "bundles", "nystrom", f"{name}_nystrom_M4096.pt")
        rff_path = os.path.join(out, "bundles", "eval_rff", f"{name}.pt")
        sw_tr_path = os.path.join(out, "bundles", "eval_sw", f"{name}_train.pt")
        sw_val_path = os.path.join(out, "bundles", "eval_sw", f"{name}_val.pt")
        is_train_enc = name in train_names
        if skip_existing and os.path.exists(rff_path) and os.path.exists(sw_val_path) \
                and (not is_train_enc or os.path.exists(nys_path)):
            logger.info("[refprep] %s: artifacts present, skipping", name)
            continue

        logger.info("[refprep] %s: extracting train features ...", name)
        train_feats = extract_features(spec, train_ds, batch_size=extract_bs,
                                       num_workers=num_workers, device=device)
        sigma_path = os.path.join(out, "sigma", f"{name}.pt")
        sigma = cache_median_sigma(train_feats, sigma_path)
        logger.info("[refprep] %s: train=%s sigma=%.4f", name, tuple(train_feats.shape), sigma)

        logger.info("[refprep] %s: extracting val features ...", name)
        val_feats = extract_features(spec, val_ds, batch_size=extract_bs,
                                     num_workers=num_workers, device=device)

        # save raw pools (resumability / re-deriving banks without re-extracting)
        save_features(train_feats, os.path.join(out, "pools", f"{name}_train.pt"))
        save_features(val_feats, os.path.join(out, "pools", f"{name}_val.pt"))

        # ---- training Nystrom reference + PID floor (10 training encoders only) ----
        if is_train_enc:
            ref_pool = os.path.join(out, "pools", f"{name}_train.pt")
            build_one(ref_pool, sigma_path, nys_path, n_landmarks=n_landmarks)
            bundle = torch.load(nys_path, map_location="cpu", weights_only=False)
            floor = compute_floor(val_feats, bundle, device=device)
            _write_floor_part(out, "pid", name, floor)
            logger.info("[refprep] %s: nystrom bundle + PID floor %.5f", name, floor)

        # ---- eval banks (all 14 encoders) ----
        sub = train_feats[:eval_subsample]
        bank = build_rff_bank(sub, sigma, rff_path, device=device)
        from ..eval.mmd_r14 import rff_mmd2
        mmd_floor = rff_mmd2(val_feats[:eval_subsample], bank["W"].float().to(device),
                             bank["b"].float().to(device), bank["mu_r"].float().to(device))
        _write_floor_part(out, "mmdr14", name, mmd_floor)
        build_sw_bank(train_feats, sw_tr_path)
        build_sw_bank(val_feats, sw_val_path)
        logger.info("[refprep] %s: eval RFF + SW banks; MMDr14 floor %.4f", name, mmd_floor)

        del train_feats, val_feats  # free before the next encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _rebuild_floor_json(out, "pid", "imagenet_val_floors.json")
    _rebuild_floor_json(out, "mmdr14", "mmdr14_val_floors.json")
    logger.info("[refprep] imagenet reference done -> %s", out)


# --------------------------------------------------------------------------------------
# flux joint reference
# --------------------------------------------------------------------------------------
def run_joint(coco_pairs: str, out: str, encoders: list[str], *, device: str = "cuda",
              img_size: int = 512, extract_bs: int = 64, num_workers: int = 8,
              sigma_scale: float = 0.25, n_landmarks: int = 4096,
              max_images: int | None = None, skip_existing: bool = False) -> None:
    """Build the FLUX joint reference: tau(c) text table + per-encoder joint Nystrom bundles."""
    from ..data.coco import CocoPairDataset, load_coco_pairs
    from ..representation.text_encoder import encode_captions

    captions = load_coco_pairs(coco_pairs)["captions"]
    if max_images:
        captions = captions[:max_images]
    tau_path = os.path.join(out, "flux2", "siglip2_text_coco.npy")
    if skip_existing and os.path.exists(tau_path):
        logger.info("[refprep] tau table present: %s", tau_path)
    else:
        logger.info("[refprep] encoding tau(c) for %d captions -> %s", len(captions), tau_path)
        os.makedirs(os.path.dirname(tau_path), exist_ok=True)
        encode_captions(captions, tau_path, device=device)

    coco_ds = CocoPairDataset(coco_pairs, img_size=img_size)
    if max_images:
        coco_ds.jpeg_paths = coco_ds.jpeg_paths[:max_images]
        coco_ds.captions = coco_ds.captions[:max_images]

    for name in encoders:
        spec = by_name(name)
        out_path = os.path.join(out, "bundles", "nystrom_joint", f"{name}_joint_M4096.pt")
        if skip_existing and os.path.exists(out_path):
            logger.info("[refprep] %s: joint bundle present, skipping", name)
            continue
        logger.info("[refprep] %s: extracting COCO image features ...", name)
        feats = extract_features(spec, coco_ds, batch_size=extract_bs,
                                 num_workers=num_workers, device=device)
        img_pool = os.path.join(out, "pools_joint", f"{name}.pt")
        sigma_path = os.path.join(out, "sigma_joint", f"{name}.pt")
        save_features(feats, img_pool)
        cache_median_sigma(feats, sigma_path)
        build_joint_one(img_pool, tau_path, sigma_path, out_path,
                        sigma_scale=sigma_scale, n_landmarks=n_landmarks)
        logger.info("[refprep] %s: joint bundle -> %s", name, out_path)
        del feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("[refprep] joint reference done. NOTE: the generator text-context pool "
                "(ctx_pool, e.g. data/fid_stats/flux2/qwen3_ctx_coco.npy) is produced by the "
                "FLUX text encoder, not here -- see docs/reproduction_map.md.")


def _resolve_encoders(arg: str | None, mode: str) -> list[str]:
    if arg:
        return [n.strip() for n in arg.split(",") if n.strip()]
    if mode == "joint":
        return [s.name for s in training_specs()]      # joint loss uses the 10 training encoders
    return [s.name for s in all_specs()]               # imagenet eval banks cover all 14


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("imagenet", help="build the ImageNet train + eval reference banks")
    pi.add_argument("--train", required=True, help="ImageNet train root (flat or class-nested)")
    pi.add_argument("--val", required=True, help="ImageNet val root")
    pi.add_argument("--out", default="data/fid_stats")
    pi.add_argument("--encoders", default=None, help="comma list (default: all 14)")
    pi.add_argument("--img-size", type=int, default=256)
    pi.add_argument("--extract-bs", type=int, default=256)
    pi.add_argument("--eval-subsample", type=int, default=DEFAULT_EVAL_SUBSAMPLE,
                    help="real-sample size for the RFF mean / MMDr14 floor")
    pi.add_argument("--max-images", type=int, default=None, help="cap images (quick test only)")
    pi.add_argument("--skip-existing", action="store_true")
    pi.add_argument("--device", default="cuda")

    pj = sub.add_parser("joint", help="build the FLUX image-text joint reference")
    pj.add_argument("--coco-pairs", required=True, help="canonical COCO pairing .npz")
    pj.add_argument("--out", default="data/fid_stats")
    pj.add_argument("--encoders", default=None, help="comma list (default: 10 training)")
    pj.add_argument("--img-size", type=int, default=512)
    pj.add_argument("--extract-bs", type=int, default=64)
    pj.add_argument("--sigma-scale", type=float, default=0.25)
    pj.add_argument("--max-images", type=int, default=None)
    pj.add_argument("--skip-existing", action="store_true")
    pj.add_argument("--device", default="cuda")
    args = ap.parse_args()

    encoders = _resolve_encoders(args.encoders, args.mode)
    if args.mode == "imagenet":
        run_imagenet(args.train, args.val, args.out, encoders, device=args.device,
                     img_size=args.img_size, extract_bs=args.extract_bs,
                     eval_subsample=args.eval_subsample, max_images=args.max_images,
                     skip_existing=args.skip_existing)
    else:
        run_joint(args.coco_pairs, args.out, encoders, device=args.device,
                  img_size=args.img_size, extract_bs=args.extract_bs,
                  sigma_scale=args.sigma_scale, max_images=args.max_images,
                  skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
