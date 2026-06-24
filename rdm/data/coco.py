"""COCO image-caption pairs for the FLUX joint reference.

The joint (text-to-image) reference couples each COCO image with one caption. This module
builds the canonical pairing (one caption per image, ordered by ``image_id``) from the
COCO ``captions_train2014.json`` annotations and saves it as a single ``.npz`` with aligned
``image_ids`` / ``captions`` / ``jpeg_paths`` arrays. Everything downstream (encoder feature
banks, text embeddings tau(c), generated pairs) is row-aligned to this ordering, so the
pairing is fixed once here.

At train time the joint reference is precomputed features + a frozen tau(c) table indexed
by caption row, so raw COCO images are read only when building the reference banks.
"""
import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .imagenet import default_transform


def build_coco_pairs(captions_json: str, image_dir: str, out_npz: str,
                     filename_template: str = "COCO_train2014_{:012d}.jpg") -> dict:
    """Build the canonical one-caption-per-image pairing and save it to ``out_npz``.

    Picks the first caption per image (deterministic), orders rows by ``image_id``, and
    records the absolute jpeg path for each. Returns the in-memory dict as well.
    """
    ann = json.load(open(captions_json))
    first: dict[int, str] = {}
    for a in ann["annotations"]:
        first.setdefault(int(a["image_id"]), str(a["caption"]).strip())
    image_ids = sorted(first)
    captions = [first[i] for i in image_ids]
    jpeg_paths = [os.path.join(image_dir, filename_template.format(i)) for i in image_ids]
    out = dict(image_ids=np.asarray(image_ids, dtype=np.int64),
               captions=np.asarray(captions, dtype=object),
               jpeg_paths=np.asarray(jpeg_paths, dtype=object))
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(out_npz, **out)
    return out


def load_coco_pairs(npz_path: str) -> dict:
    """Load the pairing ``.npz`` -> ``{image_ids, captions (list[str]), jpeg_paths (list[str])}``."""
    z = np.load(npz_path, allow_pickle=True)
    return dict(image_ids=z["image_ids"].astype(np.int64),
                captions=[str(c) for c in z["captions"]],
                jpeg_paths=[str(p) for p in z["jpeg_paths"]])


class CocoPairDataset(Dataset):
    """Row-aligned ``(image[0,1], caption_index)`` over the canonical COCO pairing."""

    def __init__(self, npz_path: str, img_size: int = 512, transform=None):
        pairs = load_coco_pairs(npz_path)
        self.jpeg_paths = pairs["jpeg_paths"]
        self.captions = pairs["captions"]
        self.transform = transform or default_transform(img_size)

    def __len__(self):
        return len(self.jpeg_paths)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.jpeg_paths[idx]).convert("RGB"))
        return img, idx   # caption index = row (use captions[idx] for the text)
