"""Report container + Table 1 / Table 7 row formatters.

A :class:`Report` wraps one model's per-encoder ratios and the panel aggregate for one
metric (SW_r14 or MMDr14). :func:`format_table` renders a markdown table over a set of
models in panel order, with the held-out encoders flagged.
"""
from dataclasses import dataclass, field

from .panel import heldout_names, panel_names


@dataclass
class Report:
    """One model's metric breakdown."""

    model: str
    metric: str                         # "SW_r14" or "MMDr14"
    aggregate: float
    per_encoder: dict = field(default_factory=dict)

    def row(self, order=None) -> str:
        """Markdown table row: ``| model | e1 | e2 | ... | aggregate |``."""
        order = order or panel_names()
        cells = [f"{self.per_encoder.get(n, float('nan')):.2f}" for n in order]
        return "| " + " | ".join([self.model, *cells, f"**{self.aggregate:.2f}**"]) + " |"


def format_table(reports: list, metric: str, order=None) -> str:
    """Render Table 1 (SW_r14) / Table 7 (MMDr14) as markdown over ``reports`` in panel order."""
    order = order or panel_names()
    held = set(heldout_names())
    header_cols = [f"{n}*" if n in held else n for n in order]   # * marks held-out
    head = "| Model | " + " | ".join(header_cols) + f" | {metric} |"
    sep = "|" + "---|" * (len(order) + 2)
    rows = [r.row(order) for r in reports]
    note = f"\n\n(* = held out from training; {metric} is the arithmetic mean over the {len(order)} encoders.)"
    return "\n".join([head, sep, *rows]) + note
