import torch

def forward():
    a = torch.ones(2)
    b = a.reshape(2)
    c = a[0:2] # So a, b, c share the same storage?

    def subfunc():
        b[0] = 2
        if b.sum() >= -1e5:
            pass

    subfunc()
    return c

print(forward()) # [2., 1.]
fn_compiled = torch.compile(forward)
print(fn_compiled()) # [1., 1.]

import torch

def forward():
    a = torch.ones((2,))
    b = a.reshape(2)
    c = a[0:2] # not b here

    def subfunc():
        b[0] = 2
        if b.sum() >= -1e5:
            pass

    subfunc()
    return c

print(forward()) # [2., 1.]
fn_compiled = torch.compile(forward)
print(fn_compiled()) # [1., 1.]

def forward(self, arg0_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    select = torch.ops.aten.select.int(arg0_1, 0, 0)
    device_put = torch.ops.prims.device_put.default(lift_fresh_copy, device(type='cuda', index=0));  lift_fresh_copy = None
    convert_element_type = torch.ops.prims.convert_element_type.default(device_put, torch.float32);  device_put = None
    select_scatter = torch.ops.aten.select_scatter.default(arg0_1, convert_element_type, 0, 0);  convert_element_type = None
    select_1 = torch.ops.aten.select.int(select_scatter, 0, 0)
    sum_1 = torch.ops.aten.sum.default(select_scatter)
    ge = torch.ops.aten.ge.Scalar(sum_1, -100000.0);  sum_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, select_scatter);  arg0_1 = select_scatter = None
    return (ge,)

from ctypes import c_void_p, c_long
import torch
import math
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_hirsheybar/gs/cgsmrhhnelkxo5fhek25abfqvgi4zsx2jkg4rrovzqbziq3mfa4z.py
# Original ATen: aten.copy, aten.ge, aten.sum, aten.select_scatter, aten.lift_fresh

# aten.copy => device_put
# aten.ge => ge
# aten.sum => sum_1
# aten.select_scatter => select_scatter
# aten.lift_fresh => lift_fresh_copy
triton_fused_copy_ge_lift_fresh_select_scatter_sum_0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import persistent_reduction
from torch._inductor.utils import instance_descriptor

@persistent_reduction(
    size_hints=[1, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp4 = tl.load(in_ptr0 + (r0), rmask, other=0)
    tmp0 = r0
    tmp1 = 0
    tmp2 = tmp0 == tmp1
    tmp3 = 2.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = -100000.0
    tmp10 = tmp8 >= tmp9
    tl.store(out_ptr1 + (0 + tl.zeros([XBLOCK, 1], tl.int32)), tmp10, None)
    tl.store(out_ptr0 + 0 + tl.zeros([XBLOCK, 1], tl.int32), tmp8, None)
''')


# kernel path: /tmp/torchinductor_hirsheybar/cd/ccdbesvbxyknip36ln46yecgrmmvdpy23y5bbt4qm3kbi2f4lhwl.py
# Original ATen: aten.copy, aten.select_scatter, aten.lift_fresh

# aten.copy => device_put
# aten.select_scatter => select_scatter
# aten.lift_fresh => lift_fresh_copy
triton_fused_copy_lift_fresh_select_scatter_1 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[2], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp4 = tl.load(in_ptr0 + (x0), xmask)
    tmp0 = x0
    tmp1 = 0
    tmp2 = tmp0 == tmp1
    tmp3 = 2.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr1 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp5, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((), (), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((), (), device='cuda', dtype=torch.bool)
        stream0 = get_cuda_stream(0)
        triton_fused_copy_ge_lift_fresh_select_scatter_sum_0.run(arg0_1, buf0, buf3, 1, 2, grid=grid(1), stream=stream0)
        triton_fused_copy_lift_fresh_select_scatter_1.run(arg0_1, arg0_1, 2, grid=grid(2), stream=stream0)
        del arg0_1
        return (buf3, )