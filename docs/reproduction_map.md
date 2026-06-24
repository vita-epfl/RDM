# Reproduction map: artifact → command → paper reference

| Paper artifact | Command | Config | Notes |
|---|---|---|---|
| **Table 1** — SW_r14 (ImageNet, iRDM 1.30) | `python reproduce.py eval-imagenet` | `configs/eval_imagenet.yaml` | render 4000+ latents → `rdm.eval.sw_r14`; N=16384, M=1024 projections |
| **Table 7** — MMDr14 (App D, iRDM 2.69) | `python reproduce.py eval-imagenet` | `configs/eval_imagenet.yaml` | RFF-MMD ratio, arithmetic mean over 14 encoders; N=50000, D=4096 |
| **Fig. 6** — PickScore preference (off-objective) | `python reproduce.py eval-imagenet` | `configs/eval_imagenet.yaml` | 4000 class-conditional latents, `"a photo of a {classname}"` (torchvision class names, offline); mean PickScore. Matched-noise paired Δ vs the step-0 baseline isolates the training effect |
| One-step ImageNet training | `GPUS=8 bash scripts/train.sh configs/imagenet.yaml` | `configs/imagenet.yaml` | pMF-H, 10 enc, Σ=10 PID, lr 1.6e-6, N=5120, 4000 steps |
| **Table 2** — FLUX GenEval (joint 0.805 / marginal 0.779) | `GPUS=8 bash scripts/train.sh configs/flux.yaml` then `python reproduce.py eval-flux` | `configs/flux.yaml` | concat joint; set `joint_enable: false` for the marginal arm. Set `load_from` in `eval_flux.yaml` to the trained student (else the base klein-4B is evaluated); each prompt's Qwen3 context is encoded on the fly |
| FLUX PickScore (21.69) | `python reproduce.py eval-flux` | `configs/eval_flux.yaml` | 499 Pick-a-Pic test prompts; context encoded per prompt, not sliced from `ctx_pool` |
| **Fig. 3** — spiral 3×6 grid | `python reproduce.py fig3` | `configs/toy_spiral.yaml` | self-contained; Nyström sharpest in every row, floor 0.033 |
| **Fig. 4 / Table 6** — batch-size axis | `python reproduce.py fig4` (low-dim) | `configs/toy_batch.yaml` | full-scale = single-encoder DINOv2 at matched wall-clock, √N lr |
| **Table 4** — distance ablation | `python reproduce.py ablation-distance` (low-dim) | `configs/ablation_distance.yaml` | order mmdx ≻ mmd_rff ≻ mmd_exact ≻ fd ≻ sw ≻ drifting |
| **Table 3** — gated vs uniform | `GPUS=8 bash scripts/train.sh configs/ablation_constrained.yaml` | `configs/ablation_constrained.yaml` | flip `pid_enable` for the uniform control |

## Reference artifacts (built once by `scripts/run_refprep.sh`)

`scripts/run_refprep.sh` wraps `python -m rdm.refprep.run imagenet --train <...> --val <...>`
(and `... joint --coco-pairs <...>` for FLUX), which builds all of the below at the paths the
configs reference. The image-feature extraction dominates; shard it across GPUs by launching
the driver with disjoint `--encoders` on different `CUDA_VISIBLE_DEVICES`.

| Artifact | Builder | Used by |
|---|---|---|
| `data/fid_stats/bundles/nystrom/<enc>_nystrom_M4096.pt` `{Z, alpha, sigma, k_rr}` (10 train enc) | `rdm.refprep.build_one` | training loss |
| `data/fid_stats/imagenet_val_floors.json` (b_phi, 10 train enc) | `rdm.refprep.compute_floor` | PID controller |
| `data/fid_stats/bundles/eval_rff/<enc>.pt` (seed 99999, 14 enc) | `rdm.refprep.build_rff_bank` | MMDr14 |
| `data/fid_stats/mmdr14_val_floors.json` (14 enc) | `rdm.eval.mmd_r14.rff_mmd2` on val | MMDr14 denominator |
| `data/fid_stats/bundles/eval_sw/<enc>_{train,val}.pt` (14 enc) | `rdm.refprep.build_sw_bank` | SW_r14 |
| `data/fid_stats/bundles/nystrom_joint/<enc>_joint_M4096.pt` | `rdm.refprep.build_joint_one` | FLUX joint loss |
| `data/fid_stats/flux2/siglip2_text_coco.npy` (τ(c)) | `rdm.representation.text_encoder.encode_captions` | FLUX joint feature |
| `data/fid_stats/flux2/qwen3_ctx_coco.npy` (generator text context over **COCO captions**) | `python scripts/build_flux2_ctx.py --captions data/coco/coco_pairs.npz --out ... --ctx-len 48` (FLUX.2 Qwen3 encoder; not refprep) | FLUX **training** conditioning. Eval contexts are encoded per eval prompt at run time, not from this pool. See `docs/flux_reference.md` |

The released **pMF-H FD-SIM** generator checkpoint is fetched by
`python scripts/download_checkpoints.py --pmfh` (linked to `checkpoints/pMF-H_FD-SIM.pth`, the
configs' `load_from`); the 14 panel encoders download lazily on first use. The FLUX.2
klein-4B / AE weights and the external `flux2` package are pointed to via `FLUX2_SRC` +
`HF_HOME` (see README).
