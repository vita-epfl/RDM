"""Reference-precompute orchestration wiring (CPU, synthetic features; no weights / no images).

Covers the parts of :mod:`rdm.refprep.run` + :func:`rdm.eval.imagenet_eval.load_eval_banks`
that don't need a GPU backbone: recursive image collection, the shard-safe floor-part merge,
and that the bank builders compose into exactly the artifacts (paths, keys, shapes) the eval
loader and ``evaluate_imagenet`` consume. The heavy image extraction is the pre-existing,
separately-exercised library path.
"""
import os
import tempfile

import torch

from rdm.eval.imagenet_eval import load_eval_banks
from rdm.eval.mmd_r14 import rff_mmd2
from rdm.refprep.build_eval_reference import build_rff_bank, build_sw_bank
from rdm.refprep.build_nystrom_reference import build_one
from rdm.refprep.compute_floors import compute_floor
from rdm.refprep.extract_features import cache_median_sigma, save_features
from rdm.refprep.run import _collect_images, _rebuild_floor_json, _resolve_encoders, _write_floor_part
from rdm.utils.io import read_json


def test_collect_images_recursive_and_sorted():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "n01", "sub"))
    for rel in ("n01/a.JPEG", "n01/sub/b.png", "c.jpg", "ignore.txt"):
        open(os.path.join(d, rel), "w").close()
    got = _collect_images(d)
    assert got == sorted(got)                                          # deterministic full-path order
    assert {os.path.basename(p) for p in got} == {"a.JPEG", "b.png", "c.jpg"}  # .txt dropped


def test_resolve_encoders_defaults():
    assert len(_resolve_encoders(None, "imagenet")) == 14     # eval banks cover all 14
    assert len(_resolve_encoders(None, "joint")) == 10        # joint loss uses the 10 trained
    assert _resolve_encoders("clip, mae ", "imagenet") == ["clip", "mae"]


def test_floor_parts_merge_is_union():
    out = tempfile.mkdtemp()
    _write_floor_part(out, "pid", "a", 0.1)
    _write_floor_part(out, "pid", "b", 0.2)            # a second "shard"
    _rebuild_floor_json(out, "pid", "imagenet_val_floors.json")
    merged = read_json(os.path.join(out, "imagenet_val_floors.json"))
    assert merged == {"a": 0.1, "b": 0.2}


def test_joint_bundle_carries_beta():
    # build_joint_one stores beta = sigma_img / s_txt; ReferencePack must surface it so the
    # FLUX coupling weights the text block correctly (not sigma_img).
    import numpy as np

    from rdm.compare.interface import ReferencePack
    from rdm.refprep.build_joint_reference import build_joint_one
    out = tempfile.mkdtemp()
    torch.manual_seed(0)
    save_features(torch.randn(600, 16), os.path.join(out, "img.pt"))
    cache_median_sigma(torch.randn(600, 16), os.path.join(out, "sig.pt"))
    np.save(os.path.join(out, "tau.npy"), torch.randn(600, 8).numpy().astype("float32"))
    b = build_joint_one(os.path.join(out, "img.pt"), os.path.join(out, "tau.npy"),
                        os.path.join(out, "sig.pt"), os.path.join(out, "joint.pt"),
                        sigma_scale=0.25, s_txt=2.0, n_landmarks=32)
    assert abs(b["beta"] - b["sigma"] / 2.0) < 1e-5            # beta = sigma_img / s_txt
    rp = ReferencePack.from_nystrom_bundle("x", torch.load(os.path.join(out, "joint.pt"),
                                           map_location="cpu", weights_only=False))
    assert rp.beta is not None and abs(rp.beta - b["beta"]) < 1e-5


def test_bank_build_to_loader_roundtrip():
    out = tempfile.mkdtemp()
    torch.manual_seed(0)
    names = ["inception", "dinov3_l"]
    floors = {}
    for n in names:
        feats, val = torch.randn(800, 32), torch.randn(400, 32)
        save_features(feats, os.path.join(out, "pools", f"{n}_train.pt"))
        sigma = cache_median_sigma(feats, os.path.join(out, "sigma", f"{n}.pt"))
        # training Nystrom bundle + PID floor (the 10-encoder path)
        nys = os.path.join(out, "bundles", "nystrom", f"{n}_nystrom_M4096.pt")
        build_one(os.path.join(out, "pools", f"{n}_train.pt"),
                  os.path.join(out, "sigma", f"{n}.pt"), nys, n_landmarks=64)
        bundle = torch.load(nys, map_location="cpu", weights_only=False)
        assert {"Z", "alpha", "sigma", "k_rr"} <= set(bundle)
        assert compute_floor(val, bundle, n_draw=200, k_draws=2, device="cpu") >= 0.0
        # eval banks (all 14)
        rff = os.path.join(out, "bundles", "eval_rff", f"{n}.pt")
        bank = build_rff_bank(feats[:400], sigma, rff, D=128, device="cpu")
        floors[n] = rff_mmd2(val[:400], bank["W"].float(), bank["b"].float(), bank["mu_r"].float())
        build_sw_bank(feats, os.path.join(out, "bundles", "eval_sw", f"{n}_train.pt"), n_samples=200)
        build_sw_bank(val, os.path.join(out, "bundles", "eval_sw", f"{n}_val.pt"), n_samples=200)
    from rdm.utils.io import write_json
    write_json(floors, os.path.join(out, "mmdr14_val_floors.json"))

    tr, va, rff_banks, fl = load_eval_banks(
        os.path.join(out, "bundles", "eval_rff"), os.path.join(out, "bundles", "eval_sw"),
        os.path.join(out, "mmdr14_val_floors.json"))
    assert set(tr) == set(va) == set(rff_banks) == set(names)
    for n in names:
        assert tr[n].ndim == 2 and va[n].ndim == 2
        assert {"W", "b", "mu_r"} <= set(rff_banks[n])
        assert fl[n] >= 0.0
