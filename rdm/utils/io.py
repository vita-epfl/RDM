"""IO helpers: image save, npz / jsonl / torch-bundle read-write.

Reference artifacts (Nystrom bundles, floors, RFF banks) are read as fp32 regardless of
the training dtype; this module centralizes those small IO conventions.
"""
import json
import os

import numpy as np
import torch


def save_uint8_png(img: torch.Tensor, path: str) -> None:
    """Save a single ``[0, 1]`` CHW float image as an 8-bit PNG."""
    from PIL import Image
    arr = (img.detach().clamp(0, 1) * 255.0).round().to("cpu", torch.uint8)
    arr = arr.permute(1, 2, 0).numpy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(arr).save(path)


def load_bundle(path: str, device: str = "cpu") -> dict:
    """Load a torch reference bundle (Nystrom / RFF / FD) and cast tensor fields to fp32."""
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        return {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in obj.items()}
    return obj


def save_bundle(obj: dict, path: str) -> None:
    """Atomically save a torch bundle (``.tmp`` then ``os.replace``)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def read_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_npz(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **arrays)


def load_npz(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
