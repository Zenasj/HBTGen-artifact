import torch

p0 = torch.tensor([[4.9334, 5.5571]])
p1 = torch.tensor([[4.5627, 5.6945]])
# torch.rand(1) # no error
# p0 = torch.rand(2) # ERROR

def fn():
    v7 = torch.cat([p0, p0], dim=0) # ERROR
    # v7 = torch.cat([p1, p1], dim=0) # ERROR
    # v7 = torch.cat([p1, p0], dim=0) # ERROR
    v1 = torch.mul(v7, v7) # v1: (5, 2)
    return v7, v1

ret_eager = fn()

compiled = torch.compile(fn)
ret_compiled = compiled()

assert torch.allclose(ret_eager[0], ret_compiled[0]), '\n'.join(map(str, ["", ret_eager[0], ret_compiled[0]]))
# ^^^ no error
assert torch.allclose(ret_eager[1], ret_compiled[1]), '\n'.join(map(str, ["", ret_eager[1], ret_compiled[1]]))
''' ^^^ WRONG!
Traceback (most recent call last):
  File "/home/colin/bug.py", line 23, in <module>
    assert torch.allclose(ret_eager[1], ret_compiled[1]), '\n'.join(map(str, ["", ret_eager[1], ret_compiled[1]]))
AssertionError: 
tensor([[24.3384, 30.8814],
        [24.3384, 30.8814]])
tensor([[0., 0.],
        [0., 0.]])
'''

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


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_kshiteej/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            out_ptr0[i0] = tmp0;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<2; i0+=1)
        {
            auto tmp0 = in_ptr1[i0];
            out_ptr1[i0] = tmp0;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=0; i0<4; i0+=1)
        {
            auto tmp0 = in_ptr2[i0];
            auto tmp1 = tmp0 * tmp0;
            out_ptr2[i0] = tmp1;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    buf2 = empty_strided((2, 2), (2, 1), device='cpu', dtype=torch.float32)
    buf0 = as_strided(buf2, (1, 2), (2, 1))  # alias
    buf1 = as_strided(buf2, (1, 2), (2, 1), 2)  # alias
    buf3 = empty_strided((2, 2), (2, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    del arg1_1
    return (buf2, buf3, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2), (2, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 2), (2, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1]))
    print(arg0_1)
    print(arg1_1)
    print(call([arg0_1, arg1_1]))