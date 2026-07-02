# Third-party components

iRDM itself is released under the MIT License (see `LICENSE`). It **vendors** (copies into this
tree) and **references** (depends on / points you at) several third-party components that carry
their own licenses. This file records provenance and attribution.

## Vendored code (copied into `rdm/`)

| Component | Files | Upstream | License |
|---|---|---|---|
| **FD-Loss networks** — the pMF-H FD-SIM denoiser, its MiT-H backbone, and shared layers. Vendored faithfully so the released `pMF-H_FD-SIM` checkpoint loads bit-exact; not retrained here. | `rdm/representation/models/pmfh_fdsim.py`, `rdm/representation/models/mit.py`, `rdm/representation/models/commons.py` | FD-Loss release (Lu et al., 2026; Yang et al., 2026) | MIT — included under iRDM's MIT license with the FD-Loss authors' permission. Please also cite the FD-Loss papers when using these networks. |
| **tf-compatible FID Inception-v3** — held-in training encoder; vendored so the 2048-d pool feature is bit-exact with the precomputed reference statistics. | `rdm/representation/backbones/inception_backbone.py` | [`toshas/torch-fidelity`](https://github.com/toshas/torch-fidelity) (weights: `weights-inception-2015-12-05-6726825d.pth`) | Apache-2.0 — retain the upstream copyright/NOTICE; the vendored code is a faithful port with iRDM-specific wrapping. |

## Referenced (not vendored — you obtain these separately)

| Component | Where | License / notes |
|---|---|---|
| **FLUX.2 `flux2` package + klein-4B / AE base weights** (Black Forest Labs) | imported by `rdm/representation/generators/flux_generator.py`; obtained via `FLUX2_SRC` + `HF_HOME` (see `docs/flux_reference.md`) | `flux2` code + FLUX.2 [klein]-4B weights under Apache-2.0. The **released iRDM FLUX student weights** (`epfl-vita/flux2-klein-1step-rdm`) are a derivative of klein-4B and inherit its terms. The AE is pulled from `black-forest-labs/FLUX.2-dev`. |
| **GenEval scorer** | invoked as an external subprocess by `rdm/eval/geneval_harness.py`; you clone it yourself | [`djghosh13/geneval`](https://github.com/djghosh13/geneval) — see its repo for license. Not bundled (heavy mmdet detector deps). |
| **Frozen encoder battery + tooling** — `timm`, `open_clip_torch`, `transformers` (SigLIP2 / Qwen3), `diffusers`, `dreamsim`, `mmdet` (GenEval detector) | ordinary pip dependencies (`pyproject.toml`) | each under its own upstream license. |

## Notes

- The **paper figures** in `figures/` are original to this work (MIT).
- If you redistribute the released checkpoints, propagate the upstream weight licenses above
  (FLUX student → Apache-2.0 klein-4B terms; pMF-H → FD-Loss terms once confirmed).
