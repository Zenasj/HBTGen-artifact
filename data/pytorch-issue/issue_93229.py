import torch

def fn(i0, i1):
    # i0: (10, 3, 10)
    # i1: (3, 10, 10)
    x1 = i0.transpose(-2, -3)
    y = torch.lerp(x1, x1, 70000)
    z = torch.lerp(i1, x1, 70000)
    return y, z

x0 = torch.rand(10, 3, 10)
x1 = torch.rand(3, 10, 10)
ret_eager = fn(x0, x1)
print('==== Eager mode OK! ====')
compiled = torch.compile(fn, fullgraph=True)
ret_compiled = compiled(x0, x1)
print('==== compile mode OK! ====')

from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


triton__0 = async_compile.triton('''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_ops.autotune import pointwise
from torch._inductor.utils import instance_descriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 300
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.libdevice.expm1(tmp0)
    tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((3, 10, 10), (10, 30, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton__0.run(arg0_1, buf0, 300, grid=grid(300), stream=stream0)
        buf1 = torch.ops.aten.lerp.Scalar(as_strided(arg0_1, (3, 10, 10), (10, 30, 1)), as_strided(arg0_1, (3, 10, 10), (10, 30, 1)), 70000)
        buf2 = buf1
        assert_size_stride(buf2, (3, 10, 10), (10, 30, 1))
        del buf1
        buf3 = torch.ops.aten.lerp.Scalar(arg1_1, as_strided(arg0_1, (3, 10, 10), (10, 30, 1)), 70000)
        del arg0_1
        del arg1_1
        buf4 = buf3
        assert_size_stride(buf4, (3, 10, 10), (10, 30, 1))
        del buf3
        return (buf0, buf2, as_strided(buf4, (10, 3, 10), (30, 10, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((10, 3, 10), (30, 10, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((3, 10, 10), (100, 10, 1), device='cuda:0', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1]))