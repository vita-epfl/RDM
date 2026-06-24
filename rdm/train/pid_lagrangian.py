"""Proportional Lagrangian controller for per-encoder weighting (Stooke et al., 2020).

A single encoder can be gamed; under fixed weights the optimizer drives the aggregate down
through whichever encoders are easiest. We instead pose the weighting as constrained
optimization: each encoder must reach its real-validation floor ``b_phi``, with its weight
the Lagrange multiplier set by proportional control under a satisfaction gate. An encoder's
excess ``e_phi = s_phi - b_phi`` (live score minus floor) decides its weight -- those at or
below floor drop out (anti-overfitting), while the violators share a fixed budget ``Sigma``
through a softmax with an adaptive temperature, so the representations farthest from real
are weighted most:

    lambda_phi  proportional to  exp( e_phi / (tau * mean_violators(e)) ),   sum = Sigma

When all encoders are satisfied the weights vanish (a natural terminal state). The live
score ``s_phi`` is the same Nystrom MMD the optimizer trains (computed on the rollout), so
floor and score are on the same scale. ``build_pid_lagrangian`` returns ``None`` when
disabled -- the uniform-weight control (the loss tail then falls back to the static weight).
"""
from __future__ import annotations

import math


class _State:
    __slots__ = ("ema", "integral", "prev_err")

    def __init__(self):
        self.ema = None
        self.integral = 0.0
        self.prev_err = None


class PIDLagrangian:
    """Gated, adaptive-temperature softmax allocation of a fixed weight budget ``total_lambda``."""

    def __init__(self, names, baselines, init_lambda=None, kp=1.0, ki=0.0, kd=0.0,
                 lambda_max=10.0, ema_beta=0.5, normalized=False, cost_clip=5.0,
                 softmax_mode=True, total_lambda=10.0, temperature=2.0):
        self.names = list(names)
        self.baselines = dict(baselines)
        self.lambdas = {n: float((init_lambda or {}).get(n, 1.0)) for n in self.names}
        self.state = {n: _State() for n in self.names}
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lambda_max, self.ema_beta = lambda_max, ema_beta
        self.normalized, self.cost_clip = normalized, cost_clip
        self.softmax_mode, self.total_lambda, self.temperature = softmax_mode, total_lambda, temperature
        self.step_count = 0

    def step(self, rollout_metric: dict) -> dict:
        """Update the per-encoder weights from the live per-encoder scores ``{name: s_phi}``."""
        z = {}
        for name in self.names:
            if name not in rollout_metric:
                continue
            st, b = self.state[name], self.baselines[name]
            s_raw = float(rollout_metric[name])
            s = s_raw if st.ema is None else (1.0 - self.ema_beta) * st.ema + self.ema_beta * s_raw
            st.ema = s
            err = (s - b) / b if self.normalized else (s - b)
            if self.cost_clip is not None:
                err = max(-self.cost_clip, min(self.cost_clip, err))
            st.integral += err
            zi = self.kp * err + self.ki * st.integral + \
                (0.0 if st.prev_err is None else self.kd * (err - st.prev_err))
            st.prev_err = err
            z[name] = zi
        if self.softmax_mode:
            active = [n for n in z if z[n] > 0.0]            # violators only
            for n in z:
                self.lambdas[n] = 0.0                        # satisfied -> 0 (anti-overfit)
            if active:
                zs = [z[n] for n in active]
                t_eff = max(self.temperature * (sum(zs) / len(zs)), 1e-12)  # adaptive temperature
                z_max = max(zs)
                exp_z = [math.exp((v - z_max) / t_eff) for v in zs]
                denom = sum(exp_z)
                for n, e in zip(active, exp_z):
                    self.lambdas[n] = float(self.total_lambda * e / denom)
        else:
            for n, zi in z.items():
                self.lambdas[n] = max(0.0, min(self.lambda_max, zi))
        self.step_count += 1
        return dict(self.lambdas)

    def get_weights(self) -> dict:
        """The current per-encoder weights ``{name: lambda}`` (consumed by ``aggregate_losses``)."""
        return dict(self.lambdas)

    def state_dict(self) -> dict:
        return {"lambdas": self.lambdas, "step_count": self.step_count,
                "state": {n: (s.ema, s.integral, s.prev_err) for n, s in self.state.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.lambdas = dict(sd["lambdas"])
        self.step_count = int(sd["step_count"])
        for n, (ema, integral, prev_err) in sd["state"].items():
            if n in self.state:
                self.state[n].ema, self.state[n].integral, self.state[n].prev_err = ema, integral, prev_err


def build_pid_lagrangian(cfg, names, baselines):
    """Construct a :class:`PIDLagrangian` from a config, or ``None`` if PID is disabled (uniform)."""
    if not getattr(cfg, "pid_enable", False):
        return None
    return PIDLagrangian(
        names, baselines,
        kp=getattr(cfg, "pid_kp", 1.0), ki=getattr(cfg, "pid_ki", 0.0), kd=getattr(cfg, "pid_kd", 0.0),
        lambda_max=getattr(cfg, "pid_lambda_max", 10.0), ema_beta=getattr(cfg, "pid_ema_beta", 0.5),
        normalized=getattr(cfg, "pid_normalized", False), cost_clip=getattr(cfg, "pid_cost_clip", 5.0),
        softmax_mode=getattr(cfg, "pid_softmax_mode", True),
        total_lambda=getattr(cfg, "pid_total_lambda", float(len(names))),
        temperature=getattr(cfg, "pid_temperature", 2.0))
