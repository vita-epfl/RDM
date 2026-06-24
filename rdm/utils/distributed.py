"""Distributed helpers: differentiable all-gather + process-group setup.

Generated images are sharded across ranks; the within-batch repulsion couples the *global*
batch, so per-rank features are gathered with a gradient-preserving all-gather and the
loss is taken on the global set (each rank's backward routes gradients to its own rows).
The model is NOT wrapped in DDP -- gradients are combined manually via this all-gather in
the GradCache middle backward plus an outer ``all_reduce`` mean.
"""
import os

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


class _DiffAllGather(torch.autograd.Function):
    """All-gather that preserves gradients for the local chunk."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        # Load-bearing: explicit stream sync before the NCCL all_gather. Without it,
        # encoder kernels launched on the default stream after a complex sampler chain in
        # bf16 autocast can race the collective (GPU 100% / mem 0% deterministic hang).
        torch.cuda.current_stream().synchronize()
        dist.all_gather(gathered, tensor.contiguous())
        gathered[ctx.rank] = tensor  # preserve local autograd graph
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunk = ctx.batch_size
        return grad_output[ctx.rank * chunk:(ctx.rank + 1) * chunk].contiguous()


@torch.compiler.allow_in_graph
def diff_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gradient-preserving all-gather; a no-op on a single process.

    ``allow_in_graph`` makes torch.compile treat this as a black box (custom autograd
    Functions are mistraced through compile otherwise).
    """
    if not is_dist():
        return tensor
    return _DiffAllGather.apply(tensor)


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """In-place all-reduce average across ranks (no-op single-process)."""
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def setup_distributed() -> tuple[int, int, int]:
    """Init the process group from torchrun env vars. Returns ``(rank, world_size, local_rank)``."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0
