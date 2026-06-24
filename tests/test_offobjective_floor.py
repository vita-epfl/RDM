"""Off-objective gate: SW_r14 / MMDr14 real-val floor ~ 1, and eval never imports the loss path.

Three independent real draws (train / val / val2): scoring val2 as the "generated" set must
give a ratio near 1 for both metrics (a fresh real draw matches real). Also a static
import-graph assertion that nothing under ``rdm.eval`` imports the training loop.
"""
import pathlib

import torch

from rdm.eval.mmd_r14 import mmd_r14, rff_mmd2
from rdm.eval.report import Report, format_table
from rdm.eval.sw_r14 import sw_r14
from rdm.refprep.build_eval_reference import build_rff_bank

torch.manual_seed(0)
D = 32
train = torch.randn(4000, D)
val = torch.randn(4000, D)
val2 = torch.randn(4000, D)


def test_sw_r14_floor_near_one():
    out = sw_r14({"e": val2, "x": val2}, {"e": train, "x": train}, {"e": val, "x": val},
                 n_proj=256, seed=0)
    assert 0.7 < out["swr14"] < 1.4, out


def test_mmd_r14_floor_near_one():
    bank = build_rff_bank(train, sigma=6.0, out_path="/tmp/irdm_test_rff.pt", D=512, device="cpu")
    floor = rff_mmd2(val, bank["W"].float(), bank["b"].float(), bank["mu_r"].float())
    out = mmd_r14({"e": val2}, {"e": bank}, {"e": floor})
    assert 0.5 < out["mmdr14"] < 1.8, out


def test_report_formats_table():
    r = Report("iRDM", "SW_r14", 1.30, {"inception": 1.27, "dinov2": 1.35})
    tbl = format_table([r], "SW_r14")
    assert "iRDM" in tbl and "SW_r14" in tbl and "dinov2*" in tbl   # held-out flagged


def test_eval_does_not_import_training_loop():
    eval_dir = pathlib.Path(__file__).resolve().parent.parent / "rdm" / "eval"
    for f in eval_dir.glob("*.py"):
        src = f.read_text()
        assert "rdm.train" not in src and "from ..train" not in src, f"{f.name} imports the loop"
