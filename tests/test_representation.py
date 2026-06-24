"""Representation-axis wiring tests (no weights / no network).

Validates the Battery facade (ordered name->(B,D), subset filter, fp32-for-logits path)
with mock backbones, the build_backbone dispatch, the autograd-safe preprocess, the
pooling primitives, and the joint concat. Backbone weight loading is exercised separately
(needs the checkpoints), but every backbone module must at least import.
"""
import importlib

import torch
import torch.nn as nn

import rdm.representation.battery as battery_mod
from rdm.representation import Battery, training_specs
from rdm.representation.backbones import build_backbone
from rdm.representation.encoder_spec import EncoderSpec, PoolType, Split
from rdm.representation.joint_feature import couple, text_weight_beta
from rdm.representation.pooling import avg_tokens, cls_token, spatial_mean
from rdm.representation.preprocess import preprocess


class _MockBackbone(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.has_logits = (spec.source == "inception")
        self.dim = spec.dim

    def forward(self, x):
        return torch.zeros(x.shape[0], self.dim)


def test_all_backbone_modules_import():
    for m in ["timm_backbone", "inception_backbone", "dreamsim_backbone",
              "radio_backbone", "flux_vae_backbone", "webssl_backbone"]:
        importlib.import_module(f"rdm.representation.backbones.{m}")


def test_build_backbone_unknown_source_raises():
    bad = EncoderSpec("x", "timm", "id", 224, PoolType.CLS, 10, Split.TRAIN)
    object.__setattr__(bad, "source", "bogus")  # bypass frozen validation to hit dispatch
    try:
        build_backbone(bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_battery_facade(monkeypatch=None):
    battery_mod.build_backbone = lambda spec, device="cpu": _MockBackbone(spec)
    bat = Battery(training_specs(), device="cpu")
    assert bat.names == [s.name for s in training_specs()]
    feats = bat(torch.rand(4, 3, 256, 256))
    assert list(feats.keys()) == bat.names               # ordered, panel order
    assert feats["inception"].shape == (4, 2048)
    assert feats["dreamsim"].shape == (4, 1792)
    sub = bat(torch.rand(2, 3, 256, 256), only={"clip", "mae"})
    assert list(sub.keys()) == ["mae", "clip"]            # still panel order, filtered


def test_preprocess_resize_normalize():
    x = torch.rand(2, 3, 256, 256)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    y = preprocess(x, mean, std, target_size=224)
    assert y.shape == (2, 3, 224, 224)
    x.requires_grad_(True)
    preprocess(x, mean, std, 224).sum().backward()
    assert x.grad is not None                              # autograd-safe


def test_pooling_primitives():
    tok = torch.randn(3, 17, 8)
    assert torch.allclose(cls_token(tok), tok[:, 0])
    assert torch.allclose(avg_tokens(tok, num_prefix_tokens=1), tok[:, 1:].mean(1))
    fmap = torch.randn(3, 8, 5, 5)
    assert torch.allclose(spatial_mean(fmap), fmap.mean(dim=[2, 3]))


def test_joint_couple():
    phi = torch.randn(4, 16, requires_grad=True)
    tau = torch.randn(4, 8)                                # frozen text rows
    beta = text_weight_beta(sigma_img=2.0, s_txt=1.0)
    Phi = couple(phi, tau, beta)
    assert Phi.shape == (4, 24)
    assert torch.allclose(Phi[:, 16:], beta * tau)
    Phi.sum().backward()
    assert phi.grad is not None                            # grad flows to image side only
