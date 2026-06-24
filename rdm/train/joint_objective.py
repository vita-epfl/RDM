"""FLUX joint-objective wiring: couple the frozen text block onto each encoder's features.

On the conditional task each encoder's image features are widened to ``[phi | beta*tau(c)]``
(:mod:`rdm.representation.joint_feature`) before the Nystrom loss, so the loss matches the
*joint* image-text law. ``beta`` is per-encoder (read from the joint bundle: ``beta =
sigma_img / s_txt``). Disabling the coupling recovers the **marginal ablation** (the
image-marginal-only objective of Table 2), bit-exact with the marginal path.
"""
from ..representation.joint_feature import couple_by_caption


class JointObjective:
    """Couple ``tau(c)`` onto per-encoder features; a no-op (marginal) when disabled."""

    def __init__(self, tau_table, betas: dict, enabled: bool = True):
        self.tau_table = tau_table       # (N_prompts, d_txt) frozen, L2-normalized
        self.betas = dict(betas)         # per-encoder text-block scale
        self.enabled = enabled

    def apply(self, feats: dict, caption_ids) -> dict:
        """Return ``{name: [phi | beta_name * tau(c)]}`` (or ``feats`` unchanged if marginal)."""
        if not self.enabled or caption_ids is None:
            return feats
        return {name: couple_by_caption(f, self.tau_table, caption_ids, self.betas.get(name, 1.0))
                for name, f in feats.items()}
