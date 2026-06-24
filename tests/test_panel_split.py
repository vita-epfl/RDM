"""Encoder-panel gate: exactly 10 train / 4 held-out, dims and pools match Table 5.

Torch-free (only imports the registry / spec). The expected values are read straight
from Table 5 of the paper so the registry cannot silently drift from the panel.
"""
from rdm.representation.encoder_spec import PoolType, Split
from rdm.representation.registry import (PANEL, all_specs, by_name, heldout_specs,
                                         training_specs)

# (name, input_res, pool, dim, split) -- Table 5, in panel order.
TABLE5 = [
    ("inception",      299, PoolType.AVG,        2048, Split.TRAIN),
    ("convnext",       224, PoolType.AVG,        1024, Split.TRAIN),
    ("mae",            224, PoolType.AVG,        1024, Split.TRAIN),
    ("clip",           256, PoolType.CLS,        1024, Split.TRAIN),
    ("dinov3_l",       224, PoolType.CLS,        1024, Split.TRAIN),
    ("pe_core_l",      224, PoolType.ATTN,       1024, Split.TRAIN),
    ("siglip2",        224, PoolType.ATTN,       1152, Split.TRAIN),
    ("aimv2_huge",     224, PoolType.AVG,        1536, Split.TRAIN),
    ("webssl_dino_1b", 224, PoolType.CLS,        1536, Split.TRAIN),
    ("dreamsim",       224, PoolType.CLS,        1792, Split.TRAIN),
    ("dinov2",         256, PoolType.CLS,        1024, Split.HELD_OUT),
    ("siglip_v1",      384, PoolType.ATTN,       1152, Split.HELD_OUT),
    ("cradiov3_l",     256, PoolType.SUMMARY,    3072, Split.HELD_OUT),
    ("flux_vae",       256, PoolType.PATCH_MEAN, 1024, Split.HELD_OUT),
]


def test_counts():
    assert len(all_specs()) == 14
    assert len(training_specs()) == 10
    assert len(heldout_specs()) == 4


def test_heldout_membership():
    assert {s.name for s in heldout_specs()} == {"dinov2", "siglip_v1", "cradiov3_l", "flux_vae"}


def test_names_unique():
    names = [s.name for s in PANEL]
    assert len(names) == len(set(names))


def test_matches_table5():
    assert [s.name for s in PANEL] == [r[0] for r in TABLE5], "panel order must match Table 1/7"
    for (name, res, pool, dim, split) in TABLE5:
        s = by_name(name)
        assert s.input_res == res, f"{name} res {s.input_res} != {res}"
        assert s.pool is pool, f"{name} pool {s.pool} != {pool}"
        assert s.dim == dim, f"{name} dim {s.dim} != {dim}"
        assert s.split is split, f"{name} split {s.split} != {split}"


def test_siglip_v1_distinct_from_siglip2():
    """The held-out SigLIP v1 (384) must not be confused with the trained SigLIP2 (256)."""
    v1, v2 = by_name("siglip_v1"), by_name("siglip2")
    assert v1.model_id != v2.model_id
    assert "v2_webli" in v2.model_id and "v2_webli" not in v1.model_id
