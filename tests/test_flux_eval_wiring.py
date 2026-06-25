"""FLUX eval-path wiring (CPU, no flux2 weights): the two fixes for the eval bugs.

1. :class:`rdm.representation.flux_text_context.Flux2TextContextEncoder` encodes prompt
   STRINGS to ``(N, ctx_len, 7680)`` -- so eval renders the actual GenEval / Pick-a-Pic
   prompts, not rows sliced from the COCO ctx_pool. We mock the external ``flux2`` Qwen3
   embedder (the heavy weights are unavailable offline) and check shape / ctx_len plumbing.
2. :func:`rdm.train.launch.load_generator_weights` loads a saved ``{"model": ...}`` student
   checkpoint for BOTH modes -- the FLUX branch previously never read ``load_from`` and so
   always evaluated the untrained klein-4B base.
"""
import os
import sys
import tempfile
import types

import torch

from rdm.representation.flux_text_context import FLUX2_QWEN3_DIM, Flux2TextContextEncoder


class _FakeEmbedder:
    """Stands in for flux2's Qwen3Embedder: returns (b, max_length, 7680) bf16."""

    def __init__(self, model_spec=None, device=None):
        self.max_length = 512
        self.model_spec = model_spec

    def __call__(self, prompts):
        return torch.zeros(len(prompts), self.max_length, FLUX2_QWEN3_DIM, dtype=torch.bfloat16)


def _install_fake_flux2():
    captured = {}

    def load_qwen3_embedder(variant="4B", device="cuda"):
        captured["variant"], captured["device"] = variant, device
        return _FakeEmbedder()

    te = types.ModuleType("flux2.text_encoder")
    te.load_qwen3_embedder = load_qwen3_embedder
    te.Qwen3Embedder = _FakeEmbedder          # explicit-model-id path bypasses load_qwen3_embedder
    sys.modules["flux2"] = types.ModuleType("flux2")
    sys.modules["flux2.text_encoder"] = te
    return captured


def _uninstall_fake_flux2():
    sys.modules.pop("flux2.text_encoder", None)
    sys.modules.pop("flux2", None)


def test_flux_text_context_encodes_prompts_at_ctxlen():
    cap = _install_fake_flux2()
    try:
        enc = Flux2TextContextEncoder(ctx_len=48, variant="4B", device="cpu")
        assert enc.embedder.max_length == 48                 # ctx_len forced onto the embedder
        out = enc.encode(["a cat", "a dog", "a bird"], batch=2)   # 2 batches (2 + 1)
        assert out.shape == (3, 48, FLUX2_QWEN3_DIM)         # one row PER PROMPT, training L
        assert out.dtype == torch.float32                    # bf16 -> fp32
        assert cap["variant"] == "4B"
    finally:
        _uninstall_fake_flux2()


def test_flux_text_context_empty_is_well_shaped():
    _install_fake_flux2()
    try:
        out = Flux2TextContextEncoder(ctx_len=16, device="cpu").encode([])
        assert out.shape == (0, 16, FLUX2_QWEN3_DIM)
    finally:
        _uninstall_fake_flux2()


def test_flux_text_context_custom_model_id_offline():
    """model_id (e.g. bf16 Qwen/Qwen3-4B for offline) goes via Qwen3Embedder, not the FP8 loader."""
    _install_fake_flux2()
    try:
        enc = Flux2TextContextEncoder(ctx_len=32, model_id="Qwen/Qwen3-4B", device="cpu")
        assert enc.embedder.model_spec == "Qwen/Qwen3-4B"   # bypassed the FP8 load_qwen3_embedder
        assert enc.embedder.max_length == 32
        assert enc.encode(["x"]).shape == (1, 32, FLUX2_QWEN3_DIM)
    finally:
        _uninstall_fake_flux2()


def test_load_generator_weights_flux_loads_student():
    from rdm.train.launch import load_generator_weights
    src = torch.nn.Linear(8, 8)
    with torch.no_grad():
        src.weight.fill_(3.14)
        src.bias.fill_(-1.0)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "step_0000150.pth")
    torch.save({"model": src.state_dict(), "step": 150}, p)
    dst = torch.nn.Linear(8, 8)                              # fresh (random) "base"
    load_generator_weights(dst, p, "flux")                  # FLUX path: direct load
    assert torch.allclose(dst.weight, src.weight) and torch.allclose(dst.bias, src.bias)


def test_load_generator_weights_noop_when_unset():
    from rdm.train.launch import load_generator_weights
    dst = torch.nn.Linear(4, 4)
    w0 = dst.weight.clone()
    load_generator_weights(dst, "", "flux")                 # empty load_from -> no-op, no crash
    assert torch.allclose(dst.weight, w0)


def test_load_generator_weights_skips_missing_file():
    """A non-empty but missing load_from warns and runs the base -- it must NOT crash."""
    from rdm.train.launch import load_generator_weights
    dst = torch.nn.Linear(4, 4)
    w0 = dst.weight.clone()
    load_generator_weights(dst, "/no/such/checkpoint_xyz.pth", "flux")
    assert torch.allclose(dst.weight, w0)


def test_build_flux2_ctx_loads_jsonl_prompts():
    """The ctx_pool builder reads prompts from a .jsonl (no flux2/GPU needed for this path)."""
    import importlib.util
    import json
    from types import SimpleNamespace
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "build_flux2_ctx", os.path.join(root, "scripts", "build_flux2_ctx.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "p.jsonl")
    with open(p, "w") as f:
        f.write(json.dumps({"prompt": "a cat", "image_id": 0}) + "\n")
        f.write(json.dumps({"prompt": "a dog", "image_id": 1}) + "\n")
    got = mod._load_prompts(SimpleNamespace(captions=None, jsonl=p, key="prompt"))
    assert got == ["a cat", "a dog"]


def test_bundled_eval_prompts_match_loaders():
    """The shipped GenEval / Pick-a-Pic assets parse with the loaders the eval uses."""
    from rdm.data.prompts import load_geneval_metadata, load_jsonl_prompts
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ge = load_geneval_metadata(os.path.join(root, "assets", "geneval_prompts.jsonl"))
    pa = load_jsonl_prompts(os.path.join(root, "assets", "pickapic_test_prompts.jsonl"))
    assert len(ge) == 553 and all("prompt" in m and "tag" in m for m in ge)
    assert len(pa) == 499 and all(isinstance(p, str) and p for p in pa)
