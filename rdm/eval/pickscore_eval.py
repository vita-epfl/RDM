"""PickScore -- a learned human-preference proxy the objective never optimizes.

PickScore (Kirstain et al., 2023; ``yuvalkirstain/PickScore_v1``, a CLIP-ViT-H fine-tuned on
Pick-a-Pic) scores image-text alignment with human preference. iRDM reports the raw mean
PickScore over the 499 Pick-a-Pic test prompts (FLUX) and a matched-noise paired win-rate against
the prior best generator (ImageNet class prompts). Off-objective: nothing here touches the
training loss.
"""
import torch


class PickScorer:
    """Frozen PickScore model + processor; ``score`` returns per-pair PickScores."""

    def __init__(self, model_id: str = "yuvalkirstain/PickScore_v1", device: str = "cuda"):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained(model_id).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def score(self, images, prompts: list[str]) -> torch.Tensor:
        """PickScore for each (image, prompt); ``images`` is a ``[0,1]`` NCHW batch.

        Gotcha: the batch is converted to **PIL uint8** first so the CLIP processor does its
        own resize+normalize. Passing ``[0,1]`` floats straight to the processor would let its
        ``do_rescale`` divide by 255 a second time and corrupt the score.
        """
        from torchvision.transforms.functional import to_pil_image
        pil = [to_pil_image(img.cpu().clamp(0, 1)) for img in images]   # float[0,1] -> uint8 PIL
        img_in = self.processor(images=pil, return_tensors="pt").to(self.device)
        txt_in = self.processor(text=prompts, padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(self.device)
        img_e = torch.nn.functional.normalize(self.model.get_image_features(**img_in), dim=-1)
        txt_e = torch.nn.functional.normalize(self.model.get_text_features(**txt_in), dim=-1)
        logit_scale = self.model.logit_scale.exp()
        return (logit_scale * (img_e * txt_e).sum(-1))   # (B,) PickScore per pair


@torch.no_grad()
def mean_pickscore(scorer: PickScorer, images, prompts: list[str], batch_size: int = 64) -> float:
    """Mean PickScore over a set of (image, prompt) pairs."""
    total, n = 0.0, 0
    for lo in range(0, len(prompts), batch_size):
        s = scorer.score(images[lo:lo + batch_size], prompts[lo:lo + batch_size])
        total += float(s.sum())
        n += s.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def head_to_head_winrate(scorer: PickScorer, images_a, images_b, prompts: list[str],
                         batch_size: int = 64) -> float:
    """Fraction of prompts on which ``images_a`` outscores ``images_b`` (matched pairs)."""
    wins, n = 0, 0
    for lo in range(0, len(prompts), batch_size):
        p = prompts[lo:lo + batch_size]
        sa = scorer.score(images_a[lo:lo + batch_size], p)
        sb = scorer.score(images_b[lo:lo + batch_size], p)
        wins += int((sa > sb).sum())
        n += sa.shape[0]
    return wins / max(n, 1)
