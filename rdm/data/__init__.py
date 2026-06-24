"""Real-image and prompt ingestion for the offline reference and evaluation.

These feed the reference precompute (:mod:`rdm.refprep`) and the FLUX joint reference, not
the training loop (the data side is frozen into the Nystrom reference). ImageNet via
:mod:`imagenet`, COCO image-caption pairs via :mod:`coco`, prompt loaders via
:mod:`prompts`, and the offline FLUX render driver in :mod:`flux_gen_driver`.
"""
from .coco import CocoPairDataset, build_coco_pairs, load_coco_pairs
from .imagenet import (ImageFolderDataset, ImageListDataset, ManifestDataset,
                       build_dataloader, center_crop_arr, default_transform)
from .prompts import (imagenet_class_prompts, load_coco_captions, load_geneval_metadata,
                      load_geneval_prompts, load_jsonl_prompts)

__all__ = ["ImageFolderDataset", "ImageListDataset", "ManifestDataset", "build_dataloader",
           "center_crop_arr", "default_transform", "build_coco_pairs", "load_coco_pairs",
           "CocoPairDataset", "load_coco_captions", "load_geneval_metadata",
           "load_geneval_prompts", "load_jsonl_prompts", "imagenet_class_prompts"]
