"""FLUX.2 [klein] 4-step -> 1-step generator wrapper.

Wraps the native Black Forest Labs FLUX.2 ``Flux2`` MM-DiT so it presents the one-step
generator contract: ``sample_images_with_grad(noise, condition, sampling_args) -> latents``,
with a companion VAE tokenizer (:class:`Flux2VAETokenizer`) that decodes those latents to
``[0, 1]`` pixels. iRDM post-trains the guidance- + step-distilled klein-4B teacher into a
one-step student by flow-matching Euler (``num_steps=1`` -> ``x = noise + (0 - t0) * pred``);
the final one-step output is the generator matched by the RDM loss.

The condition is a precomputed text-context tensor ``(B, L, d)`` (the text encoder is not
loaded here; see :mod:`rdm.data.prompts`). The latent geometry for 512px output is
``(B, 128, 32, 32)`` (16x VAE compression, 128 pixel-shuffled latent channels).

External dependency: the native ``flux2`` package (Black Forest Labs) and the klein-4B /
AE weight snapshots. Point to them with the ``FLUX2_SRC`` env var (or ``flux2_src=`` arg)
and ``HF_HOME``; nothing here hardcodes a machine path.
"""
from __future__ import annotations

import glob
import logging
import os
import sys
from typing import Any

import torch
import torch.nn as nn

from .base import Generator

logger = logging.getLogger("rdm")

# native VAE spatial compression: 8x (conv) * 2x (pixel-shuffle) = 16x; z-ch 32 -> 128
FLUX2_VAE_DOWNSAMPLE = 16
FLUX2_LATENT_CHANNELS = 128

_KLEIN4B_REPO = "models--black-forest-labs--FLUX.2-klein-4B"
_KLEIN4B_FILE = "flux-2-klein-4b.safetensors"
_AE_REPO = "models--black-forest-labs--FLUX.2-dev"   # native loader pulls the AE from the dev repo
_AE_FILE = "ae.safetensors"


def _hub_root() -> str:
    return os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def _ensure_flux2_importable(flux2_src: str | None) -> None:
    src = flux2_src or os.environ.get("FLUX2_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)


def _resolve_hf_file(repo_dir: str, filename: str) -> str:
    """Find ``<hub>/<repo_dir>/snapshots/<hash>/<filename>`` without needing the hash."""
    pattern = os.path.join(_hub_root(), repo_dir, "snapshots", "*", filename)
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"FLUX.2 weight not found: {pattern}")
    return matches[-1]


class Flux2AdapterModel(nn.Module):
    """Native FLUX.2 klein-4B MM-DiT wrapped with the one-step generator contract.

    Exposes ``in_channels=128`` and ``input_size = image_resolution // 16`` (latent spatial);
    ``parameters()`` are the trainable MM-DiT params (VAE / text encoder excluded).
    """

    def __init__(self, variant: str = "klein-4b", image_resolution: int = 512,
                 checkpoint_path: str | None = None, flux2_src: str | None = None,
                 param_dtype: torch.dtype = torch.float32, gradient_checkpointing: bool = False,
                 guidance: float = 1.0, compile_blocks: bool = False):
        super().__init__()
        if variant != "klein-4b":
            raise NotImplementedError(f"only klein-4b supported (got {variant})")
        _ensure_flux2_importable(flux2_src)
        from flux2.model import Flux2, Klein4BParams

        self.variant = variant
        self.image_resolution = int(image_resolution)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.guidance = float(guidance)
        if self.image_resolution % FLUX2_VAE_DOWNSAMPLE != 0:
            raise ValueError(f"image_resolution {self.image_resolution} must be a multiple of "
                             f"{FLUX2_VAE_DOWNSAMPLE}")

        params = Klein4BParams()
        assert params.use_guidance_embed is False, "klein-4b is guidance-distilled"
        assert params.in_channels == FLUX2_LATENT_CHANNELS
        weight_path = checkpoint_path or _resolve_hf_file(_KLEIN4B_REPO, _KLEIN4B_FILE)
        logger.info("[flux2] loading klein-4b MM-DiT from %s", weight_path)
        from safetensors.torch import load_file as load_sft

        with torch.device("meta"):
            model = Flux2(params).to(torch.bfloat16)
        sd = load_sft(weight_path, device="cpu")
        model.load_state_dict(sd, strict=True, assign=True)
        del sd
        self.model = model.to(dtype=param_dtype)

        from flux2 import sampling as _flux2_sampling
        self._batched_prc_img = _flux2_sampling.batched_prc_img
        self._batched_prc_txt = _flux2_sampling.batched_prc_txt
        self._get_schedule = _flux2_sampling.get_schedule
        from flux2.model import timestep_embedding as _ts_embed
        self._timestep_embedding = _ts_embed

        self.in_channels = FLUX2_LATENT_CHANNELS
        self.input_size = self.image_resolution // FLUX2_VAE_DOWNSAMPLE
        # compile-OUTSIDE-checkpoint (the only supported order); needs preserve_rng_state=False
        # on the checkpoint calls (klein has no dropout).
        self.compile_blocks = bool(compile_blocks)
        if self.compile_blocks:
            self._run_dit = torch.compile(self._run_dit)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _run_dit(self, x, x_ids, ctx, ctx_ids, t_vec):
        """Faithful re-implementation of ``flux2.model.Flux2.forward`` (no-ref path) with
        optional per-block gradient checkpointing."""
        m = self.model
        num_txt_tokens = ctx.shape[1]
        vec = m.time_in(self._timestep_embedding(t_vec, 256))    # klein: no guidance embed
        double_block_mod_img = m.double_stream_modulation_img(vec)
        double_block_mod_txt = m.double_stream_modulation_txt(vec)
        single_block_mod, _ = m.single_stream_modulation(vec)
        img = m.img_in(x)
        txt = m.txt_in(ctx)
        pe_x = m.pe_embedder(x_ids)
        pe_ctx = m.pe_embedder(ctx_ids)
        use_ckpt = self.gradient_checkpointing and torch.is_grad_enabled()

        for block in m.double_blocks:
            if use_ckpt:
                def _dbl(_img, _txt, _block=block):
                    o_img, o_txt, _ = _block.forward_kv_extract(
                        _img, _txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt, 0)
                    return o_img, o_txt
                img, txt = torch.utils.checkpoint.checkpoint(
                    _dbl, img, txt, use_reentrant=False, preserve_rng_state=False)
            else:
                img, txt, _ = block.forward_kv_extract(
                    img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt, 0)

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)
        for block in m.single_blocks:
            if use_ckpt:
                def _sgl(_img, _block=block):
                    o_img, _ = _block.forward_kv_extract(_img, pe, single_block_mod, num_txt_tokens, 0)
                    return o_img
                img = torch.utils.checkpoint.checkpoint(
                    _sgl, img, use_reentrant=False, preserve_rng_state=False)
            else:
                img, _ = block.forward_kv_extract(img, pe, single_block_mod, num_txt_tokens, 0)

        img = img[:, num_txt_tokens:, ...]
        return m.final_layer(img, vec)

    def sample_images_with_grad(self, noise: torch.Tensor, condition, sampling_args: dict) -> torch.Tensor:
        """Run the 1-step (or N-step) flow-matching student with autograd.

        ``noise``: ``(B, 128, H, W)`` standard-normal latent noise. ``condition``: precomputed
        text-context tensor ``(B, L, d)``. Returns AE-normalized latents ``(B, 128, H, W)``.
        """
        if not torch.is_tensor(condition):
            raise TypeError(f"condition must be a ctx tensor (B, L, d); got {type(condition).__name__}")
        B = noise.shape[0]
        if condition.shape[0] != B:
            raise ValueError(f"batch mismatch: noise B={B} vs ctx B={condition.shape[0]}")
        ctx = condition.to(device=noise.device, dtype=noise.dtype)
        x, x_ids = self._batched_prc_img(noise)
        ctx, ctx_ids = self._batched_prc_txt(ctx)
        H, W = noise.shape[-2], noise.shape[-1]
        seq_len = x.shape[1]
        num_steps = int(sampling_args.get("num_steps", 1))
        timesteps = self._get_schedule(num_steps, seq_len)
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((B,), t_curr, dtype=x.dtype, device=x.device)
            pred = self._run_dit(x, x_ids, ctx, ctx_ids, t_vec)
            x = x + (t_prev - t_curr) * pred
        from einops import rearrange
        return rearrange(x, "b (h w) c -> b c h w", h=H, w=W)


class Flux2VAETokenizer(nn.Module):
    """Native FLUX.2 AutoEncoder (frozen decoder) with the tokenizer contract.

    The adapter returns latents already in the AE's normalized space and ``decode`` applies
    its BatchNorm inverse internally, so ``normalize_z`` / ``denormalize_z`` are the identity.
    ``decode`` -> ``[-1, 1]`` pixels; ``detokenize`` -> chunked ``[0, 1]`` pixels.
    """

    def __init__(self, checkpoint_path: str | None = None, flux2_src: str | None = None,
                 torch_dtype: torch.dtype = torch.bfloat16, device=None):
        super().__init__()
        _ensure_flux2_importable(flux2_src)
        from flux2.autoencoder import AutoEncoder, AutoEncoderParams
        from safetensors.torch import load_file as load_sft

        if device is None:
            if torch.distributed.is_initialized():
                device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)
        weight_path = checkpoint_path or _resolve_hf_file(_AE_REPO, _AE_FILE)
        logger.info("[flux2-vae] loading AutoEncoder from %s", weight_path)
        with torch.device("meta"):
            ae = AutoEncoder(AutoEncoderParams())
        sd = load_sft(weight_path, device="cpu")
        ae.load_state_dict(sd, strict=True, assign=True)
        del sd
        self.vae = ae.to(device=self._device, dtype=torch_dtype).requires_grad_(False).eval()
        self.torch_dtype = torch_dtype

    def normalize_z(self, z):
        return z

    def denormalize_z(self, z):
        return z

    def decode(self, z):
        return self.vae.decode(z.to(dtype=next(self.vae.parameters()).dtype))   # (B,3,H,W) in [-1,1]

    def detokenize(self, z, decode_bsz: int | None = None):
        """Chunked decode -> ``[0, 1]`` pixels.

        NOT ``@torch.inference_mode()``: training renders through this path and needs gradients to
        flow through the (frozen) VAE back to the latent/generator. Eval callers already run under
        ``torch.no_grad``, so no graph is built there either way. (The chunked ``torch.empty`` branch
        is eval-only -- training micro-batches are ``<= decode_bsz`` and take the single-shot branch.)
        """
        if decode_bsz is None:
            pixels_per_sample = z.shape[-2] * z.shape[-1]
            decode_bsz = max(1, 64 * (32 * 32) // pixels_per_sample)
        z_bsz = z.shape[0]
        if z_bsz > decode_bsz:
            out_shape = torch.clamp(self.decode(z[:1]) * 0.5 + 0.5, 0.0, 1.0).shape
            out = torch.empty(z_bsz, *out_shape[1:], device=z.device)
            for i in range(0, z_bsz, decode_bsz):
                out[i:i + decode_bsz] = torch.clamp(self.decode(z[i:i + decode_bsz]) * 0.5 + 0.5, 0.0, 1.0)
            return out
        return torch.clamp(self.decode(z) * 0.5 + 0.5, 0.0, 1.0)


class FluxGenerator(Generator):
    """One-step FLUX.2 generator: runs the adapter then decodes via the VAE tokenizer."""

    def __init__(self, model: Flux2AdapterModel, sampling_args: dict, tokenizer: Flux2VAETokenizer,
                 args=None):
        self.model = model
        self.sampling_args = sampling_args
        self.tokenizer = tokenizer
        self.enable_amp = bool(getattr(args, "enable_amp", True))
        self.amp_dtype = getattr(args, "amp_dtype", torch.bfloat16)

    def sample(self, noise: torch.Tensor, condition: Any) -> torch.Tensor:
        with torch.autocast("cuda", enabled=self.enable_amp, dtype=self.amp_dtype):
            latent = self.model.sample_images_with_grad(noise, condition, self.sampling_args)
        return self.tokenizer.detokenize(latent)
