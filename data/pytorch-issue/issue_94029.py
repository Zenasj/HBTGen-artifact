import torch
import torch.nn as nn

import torch._dynamo.config
import torch._inductor.config
import logging
torch._inductor.config.debug = True

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        pass
    
    def forward(self, a, b, c):
        return a * b * c
        
rawmodel = encoder().cuda()

model = torch.compile(fullgraph=True, dynamic=False)(rawmodel)

a = torch.randint(10, 30, (10,))

model(a, a, a)

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
async_compile.cpp('''
#include "/tmp/torchinductor_voz/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const long* __restrict__ in_ptr0,
                       const long* __restrict__ in_ptr1,
                       const long* __restrict__ in_ptr2,
                       long* __restrict__ out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long i0=0; i0<10; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            auto tmp1 = in_ptr1[i0];
            auto tmp3 = in_ptr2[i0];
            auto tmp2 = tmp0 * tmp1;
            auto tmp4 = tmp2 * tmp3;
            out_ptr0[i0] = tmp4;
        }
    }
}
''')

async_compile.wait(globals())
del async_compile
from torch.utils.cpp_extension import load_inline
wrapper = (
'''
#include <dlfcn.h>
#include <assert.h>

template <typename KernelFunc>
KernelFunc load_cpp_kernel(const char* so_filename) {
    KernelFunc kernel_cpp;
    auto kernel_cpp_lib = dlopen(so_filename, RTLD_NOW);
    assert(kernel_cpp_lib != nullptr);
    *(void **) (&kernel_cpp) = dlsym(kernel_cpp_lib, "kernel");
    return kernel_cpp;
}
    at::Tensor call_0(std::vector<at::Tensor> args) {
    at::Tensor arg0_1, arg1_1, arg2_1;
    arg0_1 = args[0];
    arg1_1 = args[1];
    arg2_1 = args[2];
    auto buf0 = at::empty_strided({10, }, {1, }, at::ScalarType::Long); 
    static auto kernel_cpp_0 = load_cpp_kernel<void (*)(const long*,const long*,const long*,long*)>("/tmp/torchinductor_voz/bp/cbpi654e7vtdcaaovcwa75vd5kas2mcaofublsggz77tlklzvfiq.so");
    kernel_cpp_0((long*)(arg0_1.data_ptr()), (long*)(arg1_1.data_ptr()), (long*)(arg2_1.data_ptr()), (long*)(buf0.data_ptr()));
    arg0_1.reset();
    arg1_1.reset();
    arg2_1.reset();
    return buf0; }''' )

module = load_inline(
    name='inline_extension_cx2ht3bu5jda6jbylbgahi6yh2oprnq6nikbox72lgo4fqipq6c3',
    cpp_sources=[wrapper],
    functions=['call_0'],
    extra_cflags=['-std=c++17 -Wno-unused-variable -march=native -O3 -ffast-math -fno-finite-math-only -fopenmp -Wall  -D C10_USING_CUSTOM_GENERATED_MACROS'],
    extra_ldflags=['-shared -fPIC  -lgomp'],
    extra_include_paths=['-I/scratch/voz/work/pytorch/torch/include -I/scratch/voz/work/pytorch/torch/include/torch/csrc/api/include -I/scratch/voz/work/pytorch/torch/include/TH -I/scratch/voz/work/pytorch/torch/include/THC -I/data/home/voz/miniconda/envs/torch4/include/python3.10'])

def _wrap_func(f):
    def g(args):
        return f(args)
    return g
call = _wrap_func(module.call_0)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.int64)
    arg1_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.int64)
    arg2_1 = rand_strided((10, ), (1, ), device='cpu', dtype=torch.int64)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1]))