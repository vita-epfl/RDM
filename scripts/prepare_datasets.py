#!/usr/bin/env python
"""Prepare the datasets the reference precompute consumes.

ImageNet-256 is assumed already on disk as an ImageFolder (train/val) -- no fetch here; pass
its root to the refprep. For FLUX, this builds the canonical COCO image-caption pairing
(one caption per image, ordered by image id) that everything joint is row-aligned to.

    python scripts/prepare_datasets.py coco \
        --captions /path/captions_train2014.json --images /path/train2014 \
        --out data/coco/coco_pairs.npz
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coco", help="build the canonical COCO image-caption pairing")
    c.add_argument("--captions", required=True)
    c.add_argument("--images", required=True)
    c.add_argument("--out", default="data/coco/coco_pairs.npz")
    args = ap.parse_args()
    if args.cmd == "coco":
        from rdm.data.coco import build_coco_pairs
        pairs = build_coco_pairs(args.captions, args.images, args.out)
        print(f"wrote {args.out}: {len(pairs['image_ids'])} image-caption pairs")


if __name__ == "__main__":
    main()
