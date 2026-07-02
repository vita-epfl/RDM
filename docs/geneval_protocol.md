# GenEval protocol (canonical)

Every "GenEval" number in this project (checkpoint cards, `docs/reproduction_map.md`) is the
**official mmdet Mask2Former** score under the protocol below. GenEval is **scorer-dependent** —
a different detector build, config, or color classifier shifts the absolute number by ~0.01–0.025
(color tasks most of all), so pin every element here when you compare. Anchors: 4-step klein
teacher **0.794**; released 1-step student `geALLcoco s180` **0.8258**.

## Generation (`rdm.eval.geneval_harness.render_geneval`)

| item | value |
|---|---|
| prompts | the official GenEval 553 (`assets/geneval_prompts.jsonl`, = `djghosh13/geneval` `evaluation_metadata.jsonl`) |
| samples / prompt | **4** → 553 × 4 = **2212** images |
| resolution | 512² (latent 128 × 32 × 32) |
| text context | Qwen3 **ctx_len 48**, encoded per prompt (bf16 `Qwen/Qwen3-4B`; not sliced from `ctx_pool`) |
| sampling | flow-matching Euler, **1 step** (student) / 4 steps (teacher), `ckpt["model"]` |
| **noise seed** | **`46_000_000 + prompt_idx*100 + sample_idx`**, one generator **per sample** (`GENEVAL_SEED_BASE`) |
| precision | bf16 autocast |

The **per-sample seed formula is load-bearing**: using one `manual_seed(prompt_idx)` stream instead
renders different images and shifts the score by checkpoint-dependent image-variance (±~0.01) — that
was a real reproduction miss. Generation is otherwise fully deterministic → same ckpt = bit-identical
images = exactly reproducible score.

## Scoring (external — `djghosh13/geneval` official scorer)

Not bundled (heavy detector deps). Clone the scorer, point `geneval_repo` at it, and score the
render dir. The canonical build:

| element | value |
|---|---|
| detector | mmdet **3.3.0** Mask2Former **Swin-S** (COCO instance), ckpt `..._20220504_001756-c9d0c4f2.pth` |
| **detector config** | the **mmdet package builtin** `mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py` — **NOT** the geneval-repo standalone copy (its `swin-t` `_base_` inheritance is broken and skews colors) |
| color classifier | open_clip **ViT-L-14, `pretrained="openai"`** zero-shot (`clip_benchmark`) |
| thresholds | conf **0.3** · counting **0.9** · position **0.1** (GenEval defaults) |
| **Overall** | **unweighted mean of the 6 per-task correctness rates** (not the per-image mean) |
| env | mmdet 3.3.0 · mmcv 2.1.0 (sm90 source build) · open-clip-torch 3.3.0 · transformers 4.40.2 · torch 2.1–2.3 +cu121 |

Command (render dir → per-image jsonl → summary):

```bash
python evaluation/evaluate_images.py <render_dir> --outfile results.jsonl \
    --model-path <detector_dir> --model-config <mmdet_builtin_mask2former_config> \
    --options model=mask2former
python evaluation/summary_scores.py results.jsonl        # prints per-task + Overall
```

> On mmdet **3.x** the official `evaluate_images.py` needs the mmdet-3 result API
> (`result.pred_instances`) rather than the mmdet-2 tuple — use an mmdet-3-ported scorer if your
> `evaluate_images.py` still unpacks `result[0]`. On a Hopper (sm90) node mmcv needs a source build.

## Known scorer offsets (read before quoting old numbers)

- The HuggingFace `facebook/mask2former-swin-small-coco-instance` scorer reads **systematically
  ~0.025 lower** than the canonical mmdet build (different decision path), and had a COCO-vs-VOC
  class-name bug that depressed ~18% of prompts. Relative comparisons survive an offset; absolute
  numbers do **not** transfer across scorer builds or to the GenEval leaderboard.
- Using the geneval-repo **standalone** detector config instead of the mmdet builtin shifts the
  color / color_attr tasks (the color classifier runs on differently-parsed detections).

## Caveat — partial in-distribution

For `phase47+` curated/mix training the GenEval 553 prompts appear in the generator pool
(self-distillation channel, teacher-verified), so GenEval is partly **in-distribution** for those
runs; held-out compositional generalization (e.g. T2I-CompBench) is a separate, open measurement.
