"""Strictly off-objective evaluation. This package never imports the training loss path.

:mod:`sw_r14` is the primary Sliced-Wasserstein metric (eq. 5); :mod:`mmd_r14` is the
secondary kernel-MMD cross-check (App D); :mod:`panel` / :mod:`report` provide the shared
14-encoder bank and the Table 1 / Table 7 formatters; :mod:`geneval_harness` and
:mod:`pickscore_eval` cover the text-to-image evaluations.
"""
from .flux_eval import evaluate_flux
from .geneval_harness import GENEVAL_CATEGORIES, summarize_official_results
from .imagenet_eval import evaluate_imagenet
from .mmd_r14 import mmd_r14
from .panel import build_eval_battery, extract_features, heldout_names, panel_names, trained_names
from .pickscore_eval import head_to_head_winrate, mean_pickscore
from .report import Report, format_table
from .sw_r14 import sw_r14, sw_ratio_per_encoder

__all__ = ["sw_r14", "sw_ratio_per_encoder", "mmd_r14", "build_eval_battery", "extract_features",
           "panel_names", "heldout_names", "trained_names", "Report", "format_table",
           "evaluate_flux", "evaluate_imagenet", "mean_pickscore", "head_to_head_winrate",
           "summarize_official_results", "GENEVAL_CATEGORIES"]
