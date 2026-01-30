tl.store(out_ptr0, xindex)

R4 = 0x411a7319
R8 = 0x3943bb2d
R12 = 0x305439af

import triton.language as tl
import triton
from torch._inductor.triton_ops.autotune import template
from torch._inductor.utils import instance_descriptor

@triton.jit
def triton_mm(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    BLOCK_M : tl.constexpr = 32
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128

    A = arg_A
    B = arg_B

    M = 2
    N = 1280
    K = 320
    stride_am = 320
    stride_ak = 1
    stride_bk = 1
    stride_bn = 320

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (1280*idx_m)
    tmp0 = tl.load(in_ptr0 + (idx_n + tl.zeros(mask.shape, tl.int32)), mask).to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (xindex + tl.zeros(mask.shape, tl.int32)), tmp1, mask)

import torch
from torch._inductor import debug
debug=True

def fn(inp, a, b):
    return torch.addmm(inp, a, b.t())

a=torch.randn(2, 320, device="cuda", dtype=torch.half)
b=torch.randn(1280, 320, device="cuda", dtype=torch.half)
out=torch.empty(1280, device="cuda", dtype=torch.half)
print(fn(out, a, b))
opt=torch.compile(fn, mode="max-autotune")
opt(out, a, b)

from torch._inductor import config as inductor_config
inductor_config.triton.cudagraphs = True