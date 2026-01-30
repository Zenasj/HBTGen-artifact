import torch.nn as nn

kernel_cpp_4 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/zt/cztcl2vp5yqlnhofzpqfficjcxgyict6e3xhfdd7sdbkipp4p44x.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(20)
    {
        {
            #pragma omp for 
            for(long i0=0; i0<8388608; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + 16*i0);
                tmp2.store(out_ptr1 + 16*i0);
            }
            #pragma omp for simd simdlen(8) 
            for(long i0=134217728; i0<134217728; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                auto tmp2 = tmp0 * tmp1;
                out_ptr0[i0] = tmp2;
                out_ptr1[i0] = tmp2;
            }
        }
    }
}
''')

import time
from torch._inductor import config
from torch.utils import mkldnn as mkldnn_utils


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 =  torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 =  torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.silu1 = torch.nn.SiLU(inplace=True)
        self.silu2 = torch.nn.SiLU(inplace=True)

    def forward(self, x, y):
        x = self.conv1(x) + y
        x = self.silu1(x)
        identity = x
        x = self.conv2(x)
        x += identity
        return x

model = Model().eval()
print(model)

opt_model = torch._dynamo.optimize('inductor')(model)

batch_size = 128

x = torch.randn(batch_size, 64, 112, 112).to(memory_format=torch.channels_last)
y = torch.randn(batch_size, 64, 112, 112).to(memory_format=torch.channels_last)


with torch.no_grad():
    for i in range(3):
        out = opt_model(x, y)

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


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/dl/cdljpywww2h2ag4o35mwbvm45hhasxnxkhqgbupxnk3y7olula65.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1)
{
    #pragma omp parallel num_threads(40)
    {
        {
            #pragma omp for
            for(long i0=0; i0<6422528; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(out_ptr0 + 16*i0);
                tmp2.store(out_ptr1 + 16*i0);
            }
            #pragma omp for simd simdlen(8)
            for(long i0=102760448; i0<102760448; i0+=1)
            {
                auto tmp0 = in_ptr0[i0];
                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                auto tmp2 = tmp0 * tmp1;
                out_ptr0[i0] = tmp2;
                out_ptr1[i0] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    buf0 = torch.ops.mkldnn._convolution_pointwise.binary(arg4_1, arg5_1, arg0_1, arg1_1, (0, 0), (1, 1), (1, 1), 1, 'add', None, None, [], None)
    assert_size_stride(buf0, (128, 64, 112, 112), (802816, 1, 7168, 64))
    del arg0_1
    del arg1_1
    del arg4_1
    del arg5_1
    buf1 = empty_strided((128, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((128, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()))
    del buf0
    buf3 = torch.ops.mkldnn._convolution_pointwise.binary(buf1, buf2, arg2_1, arg3_1, (0, 0), (1, 1), (1, 1), 1, 'add', None, None, [], None)
    assert_size_stride(buf3, (128, 64, 112, 112), (802816, 1, 7168, 64))
    del arg2_1
    del arg3_1
    return (buf3, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 64, 1, 1), (1, 0, 0, 0), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, 64, 1, 1), (1, 0, 0, 0), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]))

kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/dl/cdljpywww2h2ag4o35mwbvm45hhasxnxkhqgbupxnk3y7olula65.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0)
{
    #pragma omp parallel num_threads(40)
    {
        {
            #pragma omp for
            for(long i0=0; i0<6422528; i0+=1)
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + 16*i0);
            }
            #pragma omp for simd simdlen(8)
            for(long i0=102760448; i0<102760448; i0+=1)
            {
                auto tmp0 = in_out_ptr0[i0];
                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                auto tmp2 = tmp0 * tmp1;
                in_out_ptr0[i0] = tmp2;
            }
        }
    }
}
''')