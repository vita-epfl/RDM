"""Eval-panel facade: one bank all evaluators pull features through.

Re-exports the registry split and builds the 14-encoder evaluation battery
(:meth:`rdm.representation.battery.Battery.for_eval`), so SW_r14 and MMDr14 score against
exactly the same features in the same Table 1 / Table 7 order. This package is strictly
off-objective and never imports the training loss path.
"""
from ..representation.battery import Battery
from ..representation.registry import all_specs, by_name, heldout_specs, training_specs


def build_eval_battery(device: str = "cuda") -> Battery:
    """The 14-encoder evaluation battery (10 train + 4 held-out)."""
    return Battery.for_eval(device)


def extract_features(battery: Battery, images, only=None) -> dict:
    """Run the eval battery over a ``[0, 1]`` image batch -> ``{name: (B, d)}`` fp32 features."""
    return {k: v.float() for k, v in battery(images, only=only).items()}


def panel_names() -> list[str]:
    """The 14 encoder names in panel order."""
    return [s.name for s in all_specs()]


def heldout_names() -> list[str]:
    return [s.name for s in heldout_specs()]


def trained_names() -> list[str]:
    return [s.name for s in training_specs()]
