"""ImageNet-256 ingestion for the offline reference precompute and evaluation.

The training loop itself is feature-bank based (it consumes the frozen Nystrom reference,
not raw ImageNet at every step); raw ImageNet images are streamed only by the reference
precompute (:mod:`rdm.refprep`) and by evaluation. This module provides the ADM-style
center crop, flat-folder / explicit-list / manifest datasets, and a DataLoader with an
optional ``DistributedSampler`` for per-rank sharding.
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchvision.transforms as transforms

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def center_crop_arr(pil_image: "Image.Image", image_size: int) -> "Image.Image":
    """ADM-style center crop to ``image_size`` (BOX downsample then BICUBIC then crop)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size),
                                     resample=Image.Resampling.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size),
                                 resample=Image.Resampling.BICUBIC)
    arr = np.array(pil_image)
    cy, cx = (arr.shape[0] - image_size) // 2, (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[cy:cy + image_size, cx:cx + image_size])


def default_transform(img_size: int = 256):
    """Center-crop to ``img_size`` and convert to a ``[0, 1]`` CHW tensor."""
    return transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor(),
    ])


class ImageFolderDataset(Dataset):
    """Flat image folder -> ``[0, 1]`` center-cropped tensors (sorted by filename)."""

    def __init__(self, folder: str, img_size: int = 256, transform=None):
        paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                 if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        if not paths:
            raise FileNotFoundError(f"no images in {folder}")
        self.paths = paths
        self.transform = transform or default_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class ImageListDataset(Dataset):
    """Dataset from an explicit list of image paths -> ``[0, 1]`` center-cropped tensors."""

    def __init__(self, paths: list[str], img_size: int = 256, transform=None):
        self.paths = paths
        self.transform = transform or default_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


class ManifestDataset(Dataset):
    """ImageNet train/val from a manifest of ``"<relpath> <class_id>"`` lines.

    ``root`` is prepended to each relative path (e.g. ``<data_path>/train``). Returns
    ``(image[0,1], label)`` so the same manifest drives both feature extraction and
    class-conditional evaluation.
    """

    def __init__(self, manifest: str, root: str, img_size: int = 256, transform=None):
        self.root = root
        self.samples = []
        with open(manifest) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                rel, label = parts[0], (int(parts[1]) if len(parts) > 1 else -1)
                self.samples.append((rel, label))
        self.transform = transform or default_transform(img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, label = self.samples[idx]
        img = self.transform(Image.open(os.path.join(self.root, rel)).convert("RGB"))
        return img, label


def build_dataloader(dataset: Dataset, batch_size: int = 64, num_workers: int = 8,
                     distributed: bool = False) -> DataLoader:
    """DataLoader with an optional (deterministic, no-shuffle) ``DistributedSampler``."""
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                      num_workers=num_workers, pin_memory=True, drop_last=False)
