import torch.nn as nn

3
import torch

import triton
import triton.language as tl

# CVMM from: https://github.com/RobertCsordas/moe_layer/blob/master/triton_src/moe_layer/cvmm.py
# Based on: https://github.com/openai/triton/blob/main/python/tutorials/03-matrix-multiplication.py

from typing import Union, Optional
from dataclasses import dataclass


@dataclass
class CVMMSel:
    raw_sel: torch.Tensor
    sel: torch.Tensor
    sel_index: torch.Tensor
    out_index: Optional[torch.Tensor] = None


def cvmm_prepare_sel(sel: torch.Tensor, n_experts: int) -> CVMMSel:
    fsel = sel.flatten()
    ssel, sel_index = fsel.sort()
    return CVMMSel(sel, ssel.view_as(sel), sel_index, None)



# !!!!!! Leaving just one autotune config solves the "RuntimeError: CUDA error: an illegal memory access was
# encountered" problem !!!!!!
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def cvmm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, index_ptr, sel_ptr, out_index_ptr,
    # Matrix dimensions
    M, N, K,
    stride_cm, stride_cn,
    stride_index, stride_sel, stride_out_index,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_n = (pid % num_pid_in_group) // group_size_m

    pid_m = first_pid_m + (pid % group_size_m)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    remap_offs_am = tl.load(index_ptr + stride_index * offs_am)

    # Create offset pointers
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)


    # !!!!!! Removing this IF solves the "RuntimeError: CUDA error: an illegal memory access was encountered" problem,
    # even though it is always False in this example !!!!!!
    # To test it, keep the else branch.
    if out_index_ptr is not None:
        remap_offs_cm = tl.load(out_index_ptr + stride_out_index * offs_am)
    else:
        remap_offs_cm = remap_offs_am

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * remap_offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



def cvmm_triton(x: torch.Tensor, sel_index: torch.Tensor, sel: torch.Tensor, keys: torch.Tensor, out_dtype: torch.dtype, out_index: Optional[torch.Tensor] = None):
    x = x.flatten(end_dim=-2)
    assert x.shape[-1] == keys.shape[1]

    sel_shape = sel.shape
    sel = sel.flatten()

    M = sel.shape[0]
    O, K, N = keys.shape
    # Allocates output.
    out = torch.empty((M, N), device=x.device, dtype=out_dtype)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    cvmm_kernel[grid](
        x, keys, out, sel_index, sel, None,
        M, N, K,
        out.stride(0), out.stride(1),
        sel_index.stride(0), sel.stride(0), 0,
    )

    return out.view(*sel_shape, N)


class CVMM(torch.autograd.Function):
    warned = False

    @staticmethod
    def forward(ctx, x: torch.Tensor, sel_index: torch.Tensor, sel: torch.Tensor, keys: torch.Tensor, out_index: Optional[torch.Tensor] = None):
        ctx.save_for_backward(x, keys, sel, sel_index, out_index)

        out_type = torch.float16 if torch.is_autocast_enabled() else x.dtype
        res = cvmm_triton(x, sel_index, sel, keys, out_type, out_index)
        ctx.op_type = out_type
        ctx.keys_type = keys.dtype
        ctx.is_autocast = torch.is_autocast_enabled()
        return res

    @staticmethod
    def backward(ctx, grad_output):
        x, keys, sel, sel_index, out_index = ctx.saved_tensors

        keys_dt = keys

        grad_x_full = cvmm_triton(grad_output, sel_index, sel, keys_dt.transpose(1,2), ctx.op_type, None)
        grad_x = grad_x_full.view_as(x)

        return grad_x, None, None, None, None


def cvmm(x: torch.Tensor, sel: Union[torch.Tensor, CVMMSel], keys: torch.Tensor):
    if not isinstance(sel, CVMMSel):
        sel = cvmm_prepare_sel(sel, keys.shape[0])

    return CVMM.apply(x, sel.sel_index, sel.sel, keys, sel.out_index)

# Compile test


class Model(torch.nn.Module):
    def forward(self, x, sel, w):
        return cvmm(x, sel, w)

model = torch.compile(Model().cuda())
# model = Model().cuda()


torch.manual_seed(0)
n_experts = 8
n_channels = 64
expert_size = 64
bs = 64

device = torch.device("cuda")
dtype = torch.float16

keys = torch.nn.Parameter(torch.randn(n_experts, n_channels, expert_size, dtype=dtype, device=device))
testvec = torch.randn(bs, n_channels, dtype=dtype, device=device)
sel = torch.randint(0, n_experts, (bs,), dtype=torch.int32, device=device)

print(model(testvec, sel, keys).shape)