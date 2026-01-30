import torch.nn as nn

from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_fsp/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0)
{
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for 
        for(long i0=0; i0<16; i0+=1)
        {
            #pragma GCC ivdep
            for(long i1=0; i1<4608; i1+=1)
            {
                {
                    {
                        auto tmp0 = in_ptr0[i1 + (4608*i0)];
                        out_ptr0[i0 + (16*i1)] = tmp0;
                    }
                }
            }
        }
    }
}
''')


kernel_cpp_1 = async_compile.cpp('''
#include "/tmp/torchinductor_fsp/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    #pragma omp parallel num_threads(6)
    {
        #pragma omp for 
        for(long i0=0; i0<36864; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 8*i0);
            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
            tmp1.store(in_out_ptr0 + 8*i0);
        }
        #pragma omp for simd simdlen(4) 
        for(long i0=294912; i0<294912; i0+=1)
        {
            auto tmp0 = in_out_ptr0[i0];
            auto tmp1 = tmp0 * (tmp0>0);
            in_out_ptr0[i0] = tmp1;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    buf0 = empty_strided((1, 16, 64, 72), (73728, 1, 1152, 16), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(arg2_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg2_1
    buf1 = torch.ops.mkldnn._convolution_pointwise(buf0, arg0_1, arg1_1, (1, 1), (1, 1), (1, 1), 1, 'none', [], '')
    assert_size_stride(buf1, (1, 64, 64, 72), (294912, 1, 4608, 64))
    del arg0_1
    del arg1_1
    del buf0
    buf2 = torch.ops.aten.pixel_shuffle.default(buf1, 2)
    del buf1
    buf3 = buf2
    assert_size_stride(buf3, (1, 16, 128, 144), (294912, 18432, 144, 1))
    del buf2
    buf4 = buf3; del buf3  # reuse
    kernel_cpp_1(c_void_p(buf4.data_ptr()))
    return (buf4, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 16, 3, 3), (1, 0, 0, 0), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 16, 64, 72), (73728, 4608, 72, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1]))