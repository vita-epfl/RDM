"""Generator / model smoke tests (no weights, CPU).

Instantiates the small pMF_T variant to exercise the vendored MiT + commons + pMF denoiser
architecture and its one-step ``sample_images_with_grad`` end to end, wraps it in
PMFHGenerator, and import-checks the FLUX module (whose weights/deps are external).
"""
import importlib

import torch

from rdm.representation.generators import PMFHGenerator, build_generator
from rdm.representation.models import (convert_pmf_checkpoint, pMFDenoiser_models)


def test_pmf_models_registry():
    assert "pMF_H" in pMFDenoiser_models
    # pMF_H is MiT_H / bottleneck 256 (the released FD-SIM variant)


def test_pmf_t_one_step_sample_and_grad():
    torch.manual_seed(0)
    model = pMFDenoiser_models["pMF_T"](img_size=64, num_classes=8)  # tiny, CPU-friendly
    model.train()
    z = torch.randn(2, 3, 64, 64, requires_grad=True)
    y = torch.randint(0, 8, (2,))
    x = model.sample_images_with_grad(z, y, sampling_args={"num_steps": 1, "cfg": 1.0})
    assert x.shape == (2, 3, 64, 64)
    x.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert any(p.grad is not None for p in model.parameters())


def test_pmfh_generator_wrap():
    model = pMFDenoiser_models["pMF_T"](img_size=64, num_classes=8)

    class _Args:
        enable_amp = False  # CPU
    gen = build_generator("imagenet", model, {"num_steps": 1, "cfg": 1.0}, args=_Args())
    assert isinstance(gen, PMFHGenerator)
    out = gen.sample(torch.randn(2, 3, 64, 64), torch.randint(0, 8, (2,)))
    assert out.shape == (2, 3, 64, 64)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0   # clamped to [0,1]


def test_convert_pmf_checkpoint_keys():
    sd = {"net.block._flax_linear.weight": torch.zeros(2, 2),
          "net.class_tokens": torch.zeros(1, 8, 4),
          "net.rope_freqs": torch.zeros(4)}
    out = convert_pmf_checkpoint(sd)
    assert "net.block.linear.weight" in out           # flax-style rename
    assert out["net.class_tokens"].shape == (8, 4)    # (1,N,D) -> (N,D) squeeze
    assert "net.rope_freqs" not in out                # rope buffer skipped


def test_flux_module_imports():
    importlib.import_module("rdm.representation.generators.flux_generator")  # no instantiation
