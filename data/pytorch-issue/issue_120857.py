from typing import List

import torch


E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
EPS = 1e-12

SIZES = [torch.Size([1152, 768]), torch.Size([384, 768])]


def to_fp8_saturated(x, float8_dtype: torch.dtype):
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    else:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    return x.to(float8_dtype)


def inner_fn(weights: List[torch.Tensor]):
    # All-reduce partial amaxes computed from sharded weights
    abs_weights = torch._foreach_abs(weights)
    partial_amax_tensor = abs_weights[0].new_empty(len(abs_weights))
    for i, abs_weight in enumerate(abs_weights):
        torch.max(abs_weight, out=partial_amax_tensor[i])
    # NOTE: The error reproduces without this all-reduce, so let us use a
    # single-GPU repro.
    # replicated_amax_tensor = torch.distributed._functional_collectives.all_reduce(
    #     partial_amax_tensor, "MAX", list(range(torch.distributed.get_world_size()))
    # )
    replicated_amax_tensor = partial_amax_tensor

    # Compute scales from replicated amaxes and the fp8 bit datas
    clamped_tensor = torch.clamp(replicated_amax_tensor, EPS)
    scales_tensor = E4M3_MAX_POS / clamped_tensor
    datas = []
    scales = []
    for i, weight in enumerate(weights):
        scale = scales_tensor[i]
        weight_scaled = weight * scale
        datas.append(to_fp8_saturated(weight_scaled, torch.float8_e4m3fn))
        scales.append(scale)
    return datas, scales


weights = [torch.randn(size, device="cuda") for size in SIZES]
torch.compile(inner_fn)(weights)

import torch
from torch._inductor import config

torch.set_default_device("cuda")

config.benchmark_kernel = True

@torch.compile
def f(a, b):
    a, b = torch._foreach_abs([a, b]) # fail
    # a, b = a.abs(), b.abs() # works
    return a.max(), b.max()

N = 64
f(torch.randn(N, N), torch.randn(N, N))
print("bye")