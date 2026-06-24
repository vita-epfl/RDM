"""Prompt loaders: FLUX training prompts, GenEval, Pick-a-Pic, ImageNet class prompts.

Thin readers that return plain ``list[str]`` (or, for GenEval, the per-prompt metadata
dicts its scorer needs). The FLUX training prompts are the COCO captions of the joint
pairing (:mod:`rdm.data.coco`); GenEval and Pick-a-Pic drive text-to-image evaluation;
the ImageNet class prompts are the ``"a photo of a {class}"`` strings used for the
class-conditional PickScore.
"""
import json

import numpy as np

IMAGENET_PROMPT_TEMPLATE = "a photo of a {}"


def load_coco_captions(npz_path: str) -> list[str]:
    """FLUX training prompts = the COCO captions of the canonical pairing."""
    from .coco import load_coco_pairs
    return load_coco_pairs(npz_path)["captions"]


def load_geneval_metadata(jsonl_path: str) -> list[dict]:
    """GenEval per-prompt metadata dicts (line order preserved; each has a ``prompt`` key)."""
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_geneval_prompts(jsonl_path: str) -> list[str]:
    """Just the GenEval prompt strings, in file order."""
    return [m["prompt"] for m in load_geneval_metadata(jsonl_path)]


def load_jsonl_prompts(jsonl_path: str, key: str = "prompt") -> list[str]:
    """Generic ``.jsonl`` prompt reader (Pick-a-Pic test prompts, etc.)."""
    out = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line)[key])
    return out


def imagenet_class_prompts(class_names: list[str],
                           template: str = IMAGENET_PROMPT_TEMPLATE) -> list[str]:
    """Map a list of ImageNet class names to ``"a photo of a {class}"`` prompts."""
    return [template.format(name) for name in class_names]


def imagenet_class_prompts_torchvision(template: str = IMAGENET_PROMPT_TEMPLATE) -> list[str]:
    """The 1000 ImageNet class-name prompts in class-index (sorted-wnid) order.

    Class names come from ``torchvision``'s bundled ``ResNet50_Weights`` metadata (offline, no
    download), matching the generator's class-conditioning index. This is the prompt set the
    off-objective ImageNet PickScore (Fig. 6) uses, so no external asset is needed.
    """
    from torchvision.models import ResNet50_Weights
    names = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]   # 1000 names, class-index order
    return imagenet_class_prompts(list(names), template)


def build_imagenet_class_prompts(class_names: list[str], out_json: str,
                                 template: str = IMAGENET_PROMPT_TEMPLATE) -> list[str]:
    """Materialize ``assets/imagenet_class_prompts.json`` from a 1000-entry class-name list."""
    prompts = imagenet_class_prompts(class_names, template)
    with open(out_json, "w") as f:
        json.dump(prompts, f, indent=2)
    return prompts


def load_prompts_npz(npz_path: str, key: str = "prompts") -> list[str]:
    """Load a materialized prompt array saved as ``.npz``."""
    return [str(p) for p in np.load(npz_path, allow_pickle=True)[key]]
