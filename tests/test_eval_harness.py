"""GenEval / PickScore harness logic (mock-based; no detector / no CLIP weights).

Tests the aggregation that turns raw scorer output into the reported numbers: the GenEval
per-category + overall summary, the PickScore mean and head-to-head win-rate, and that the
FLUX/ImageNet eval orchestrators import.
"""
import importlib

import torch

from rdm.eval.geneval_harness import GENEVAL_CATEGORIES, summarize_official_results
from rdm.eval.pickscore_eval import head_to_head_winrate, mean_pickscore
from rdm.utils.io import write_jsonl


class _MockScorer:
    """PickScorer stub: score = sum of pixels + a per-set bias (ignores prompts)."""

    def __init__(self, bias=0.0):
        self.bias = bias

    def score(self, images, prompts):
        return images.reshape(images.shape[0], -1).mean(1) + self.bias


def test_geneval_categories():
    assert len(GENEVAL_CATEGORIES) == 6 and "color_attr" in GENEVAL_CATEGORIES


def test_geneval_summary(tmp_path="/tmp/irdm_geneval_test.jsonl"):
    rows = [{"tag": "colors", "correct": True}, {"tag": "colors", "correct": False},
            {"tag": "counting", "correct": True}, {"tag": "single_object", "correct": True}]
    write_jsonl(rows, tmp_path)
    s = summarize_official_results(tmp_path)
    assert abs(s["colors"] - 0.5) < 1e-9
    assert abs(s["counting"] - 1.0) < 1e-9
    assert abs(s["overall"] - 0.75) < 1e-9


def test_pickscore_mean_and_winrate():
    a = torch.ones(4, 3, 8, 8)          # higher mean pixel
    b = torch.zeros(4, 3, 8, 8)
    prompts = ["x"] * 4
    assert mean_pickscore(_MockScorer(), a, prompts) > mean_pickscore(_MockScorer(), b, prompts)
    assert head_to_head_winrate(_MockScorer(), a, b, prompts) == 1.0     # a always preferred
    assert head_to_head_winrate(_MockScorer(), b, a, prompts) == 0.0


def test_eval_orchestrators_import():
    importlib.import_module("rdm.eval.flux_eval")
    importlib.import_module("rdm.eval.imagenet_eval")


def test_imagenet_class_prompts_offline():
    # Fig. 6 PickScore prompts: 1000 names from torchvision (offline), class-index order.
    from rdm.data.prompts import imagenet_class_prompts_torchvision
    p = imagenet_class_prompts_torchvision()
    assert len(p) == 1000 and p[0] == "a photo of a tench"
