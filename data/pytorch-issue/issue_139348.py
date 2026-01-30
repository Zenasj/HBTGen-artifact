import numpy as np

import torch

def fn(x):
    return x.cumsum(0)

x = torch.rand(100, 4000, device='cuda')

breakpoint()
expect = fn(x)
actual = torch.compile(fn)(x)

torch.testing.assert_allclose(expect, actual)

import torch

import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton.jit
def triton_red_fused_cumsum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4000
    rnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3 = tl.full([XBLOCK, 1], float('nan'), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4000*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4, = tl.associative_scan((tmp2,), 1, _triton_helper_fn_add0)
        tmp5 = triton_helpers.select_one((tmp4), rbase == (RBLOCK - 1), dim=-1, keep_dims=True)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp3 + tmp4
        tmp8 = tl.where(roffset > 0, tmp7, tmp4)
        tmp3 = tl.where(roffset > 0, tmp6, tmp5)
        tl.store(out_ptr0 + (x0 + (4000*r1)), tmp8, rmask & xmask)

inp = torch.rand(100, 4000, device="cuda")
out = torch.empty(100, 4000, device="cuda")

XBLOCK = 64
RBLOCK = 8

triton_red_fused_cumsum_0[(4000,)](inp, out, 4000, 100, XBLOCK, RBLOCK, debug=True)

torch.testing.assert_close(out, inp.cumsum(0))