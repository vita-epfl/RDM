"""End-to-end training-step integration test (CPU, no encoder weights).

Wires the real pMF_T generator into the Trainer with a *mock* battery (fixed linear maps of
the generated image, so gradients still flow generator -> features -> loss) and toy Nystrom
reference packs. Exercises _make_chunks -> _encode (generate + embed) -> GradCache two-pass
-> per-encoder MMD-Nystrom + self-norm + weights -> AdamW step, and checks the generator
actually updates. Also checks the PID controller end to end.
"""
import torch

from rdm.compare.interface import ReferencePack
from rdm.compare.reference_math import build_nystrom_reference
from rdm.representation.generators import PMFHGenerator
from rdm.representation.models import pMFDenoiser_models
from rdm.train.data import ClassConditioner
from rdm.train.pid_lagrangian import PIDLagrangian
from rdm.train.trainer import Trainer


class _MockBattery:
    """Fixed linear maps of the flattened generated image -> (B, d) per encoder."""

    def __init__(self, names, in_dim, d=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = {n: torch.randn(in_dim, d, generator=g) for n in names}

    def __call__(self, images, only=None):
        flat = images.reshape(images.shape[0], -1)
        names = only or self.W.keys()
        return {n: flat @ self.W[n] for n in names if n in self.W}


def _toy_refpack(name, d=16, seed=1):
    pool = torch.randn(2000, d, generator=torch.Generator().manual_seed(seed))
    b = build_nystrom_reference(pool, 1.5, n_landmarks=32, fit_n=2000, kmeans_iters=4, krr_n=2000)
    return ReferencePack.from_nystrom_bundle(name, b)


def _build_trainer(pid=None):
    torch.manual_seed(0)
    model = pMFDenoiser_models["pMF_T"](img_size=32, num_classes=4)

    class _A:
        enable_amp = False
    gen = PMFHGenerator(model, {"num_steps": 1, "cfg": 1.0}, args=_A())
    names = ["enc1", "enc2"]
    refs = {n: _toy_refpack(n, seed=i + 1) for i, n in enumerate(names)}
    battery = _MockBattery(names, in_dim=3 * 32 * 32, d=16)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    trainer = Trainer(gen, battery, refs, opt, ClassConditioner(4),
                      rollout_size=4, batch_size=2, grad_accum=2, noise_shape=(3, 32, 32),
                      grad_clip=2.0, gen_chunk=10_000, pid=pid, device="cpu")
    return trainer, model


def test_training_step_updates_generator():
    trainer, model = _build_trainer()
    before = [p.detach().clone() for p in trainer._gen_params]
    logs = trainer.step()
    assert "loss" in logs and torch.isfinite(torch.tensor(logs["loss"]))
    assert set(logs["raw_scores"]) == {"enc1", "enc2"}
    after = list(trainer._gen_params)
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), "generator did not update"


def test_training_with_pid():
    pid = PIDLagrangian(["enc1", "enc2"], baselines={"enc1": 0.0, "enc2": 0.0},
                        total_lambda=2.0, softmax_mode=True)
    trainer, _ = _build_trainer(pid=pid)
    for _ in range(2):
        logs = trainer.step()
    w = pid.get_weights()
    assert set(w) == {"enc1", "enc2"} and all(v >= 0 for v in w.values())
