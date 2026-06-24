"""The fourteen-encoder battery as data (Table 5).

Ten encoders supervise training; four are held out for evaluation only as a
generalization check. Each row is the frozen released checkpoint, read out as one
pooled image embedding ``phi(x)`` at the listed input resolution, with **no feature
normalization**. The panel deliberately spans training paradigms (supervised, SSL
distillation, masked reconstruction, language supervision, multi-teacher agglomeration,
multimodal autoregression, human-similarity tuning, a generative autoencoder) so the
representations fail in different ways.

The conceptual pool (Table 5 ``Pool`` column) is encoded here; the backbone for each
family applies it -- e.g. a CNN's ``AVG`` is a spatial mean, a ViT's ``CLS`` is token-0,
``ATTN`` is the model's attention-pool head. See
:mod:`rdm.representation.backbones` and :mod:`rdm.representation.battery`.

Note ``siglip_v1`` (held out) is the *v1* So400m/14 at 384, distinct from the trained
SigLIP2 So400m/16 at 256 -- do not collapse the two.
"""
from .encoder_spec import EncoderSpec, PoolType, Split

P, S = PoolType, Split

#: The canonical 14-encoder panel (10 train + 4 held-out), in Table 1 / Table 7 order.
PANEL: list[EncoderSpec] = [
    # ---- Training panel (10) ----
    EncoderSpec("inception",      "inception", "",                                          299, P.AVG,        2048, S.TRAIN),
    EncoderSpec("convnext",       "timm",      "convnextv2_base.fcmae_ft_in22k_in1k",       224, P.AVG,        1024, S.TRAIN),
    EncoderSpec("mae",            "timm",      "vit_large_patch16_224.mae",                 224, P.AVG,        1024, S.TRAIN),
    EncoderSpec("clip",           "timm",      "vit_large_patch14_clip_224.openai",         256, P.CLS,        1024, S.TRAIN),
    EncoderSpec("dinov3_l",       "timm",      "vit_large_patch16_dinov3.lvd1689m",         224, P.CLS,        1024, S.TRAIN),
    EncoderSpec("pe_core_l",      "timm",      "vit_pe_core_large_patch14_336.fb",          224, P.ATTN,       1024, S.TRAIN),
    EncoderSpec("siglip2",        "timm",      "vit_so400m_patch16_siglip_256.v2_webli",    224, P.ATTN,       1152, S.TRAIN),
    EncoderSpec("aimv2_huge",     "timm",      "aimv2_huge_patch14_224.apple_pt",           224, P.AVG,        1536, S.TRAIN),
    EncoderSpec("webssl_dino_1b", "webssl",    "facebook/webssl-dino1b-full2b-224",         224, P.CLS,        1536, S.TRAIN),
    EncoderSpec("dreamsim",       "dreamsim",  "",                                          224, P.CLS,        1792, S.TRAIN),
    # ---- Held out for evaluation (4) ----
    EncoderSpec("dinov2",         "timm",      "vit_large_patch14_dinov2.lvd142m",          256, P.CLS,        1024, S.HELD_OUT),
    EncoderSpec("siglip_v1",      "timm",      "vit_so400m_patch14_siglip_384.webli",       384, P.ATTN,       1152, S.HELD_OUT),
    EncoderSpec("cradiov3_l",     "radio",     "nvidia/C-RADIOv3-L",                        256, P.SUMMARY,    3072, S.HELD_OUT),
    EncoderSpec("flux_vae",       "flux_vae",  "",                                          256, P.PATCH_MEAN, 1024, S.HELD_OUT),
]

_BY_NAME = {s.name: s for s in PANEL}


def all_specs() -> list[EncoderSpec]:
    """All 14 encoder specs, in panel order."""
    return list(PANEL)


def training_specs() -> list[EncoderSpec]:
    """The 10 encoders that supervise training."""
    return [s for s in PANEL if s.split is Split.TRAIN]


def heldout_specs() -> list[EncoderSpec]:
    """The 4 held-out (evaluation-only) encoders."""
    return [s for s in PANEL if s.split is Split.HELD_OUT]


def by_name(name: str) -> EncoderSpec:
    """Look up a spec by its short name."""
    return _BY_NAME[name]
