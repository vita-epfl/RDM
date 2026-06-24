"""Paper-number / ordering fixtures (regression anchors for the published results).

Asserts the published *orderings* (which are what the method claims), not just the numbers:
the joint FLUX objective beats both its marginal ablation and the four-step teacher on
GenEval; iRDM's SW_r14 is below every released generator and above the real floor of 1;
MMDr14 preserves the ordering; the spiral floor is 0.033 and Nystrom is sharpest.
"""
from rdm.toy import EXPECTED, METHODS
from rdm.toy.run_spiral_grid import ROWS

# Published headline numbers (Tables 1, 2, 7; Figs 3, 6).
PAPER = {
    "swr14_irdm": 1.30, "swr14_floor": 1.00, "swr14_prev_sota": 2.05,
    "mmdr14_irdm": 2.69,
    "geneval_joint": 0.805, "geneval_marginal": 0.779, "geneval_4step": 0.794,
    "pickscore_irdm": 21.69, "pickscore_4step": 21.85, "pickscore_winrate_fdsim": 0.712,
    "spiral_floor": 0.033,
}


def test_swr14_orderings():
    assert PAPER["swr14_irdm"] > PAPER["swr14_floor"]            # a gap to real remains
    assert PAPER["swr14_irdm"] < PAPER["swr14_prev_sota"]        # new one-step state of the art


def test_geneval_joint_beats_marginal_and_teacher():
    assert PAPER["geneval_joint"] > PAPER["geneval_marginal"]    # joint coupling carries the gain
    assert PAPER["geneval_joint"] > PAPER["geneval_4step"]       # surpasses the 4-step teacher


def test_pickscore_close_to_teacher_and_winrate():
    assert PAPER["pickscore_irdm"] <= PAPER["pickscore_4step"]
    assert PAPER["pickscore_winrate_fdsim"] > 0.5               # preferred over the prior best


def test_spiral_floor_and_nystrom_sharpest():
    assert PAPER["spiral_floor"] == 0.033
    for bs in ROWS:
        assert EXPECTED[(bs, "nystrom")][1] == min(EXPECTED[(bs, m)][1] for m in METHODS)
