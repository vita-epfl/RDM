"""Training system: the loop, gradient caching, the PID controller, and references.

:class:`trainer.Trainer` is the central loop; :func:`grad_caching.gradcache_backward` is the
batch-coupled exact full-batch gradient; :class:`pid_lagrangian.PIDLagrangian` balances the
encoder battery; :mod:`references` loads the frozen Nystrom artifacts; :mod:`data` provides
the fresh on-policy samplers; :mod:`joint_objective` wires the FLUX joint coupling.
"""
from .grad_caching import gradcache_backward
from .joint_objective import JointObjective
from .pid_lagrangian import PIDLagrangian, build_pid_lagrangian
from .references import load_floors, load_reference_packs, load_text_table
from .trainer import Trainer

__all__ = ["Trainer", "gradcache_backward", "PIDLagrangian", "build_pid_lagrangian",
           "JointObjective", "load_reference_packs", "load_floors", "load_text_table"]
