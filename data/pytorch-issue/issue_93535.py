import torch.nn as nn

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config


torch._functorch.config.debug_partitioner = True


isolate_fails_code_str = None



# torch version: 2.1.0a0+git2c3aa09
# torch cuda version: None
# torch git version: 2c3aa09cde00fe4b45307770ebc86178d61773e7


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', torch.randn([8, 1, 384], dtype=torch.float32))
        self.register_buffer('_tensor_constant1', torch.randn([8, 196, 384], dtype=torch.float32))

    
    
    def forward(self):
        _tensor_constant0 = self._tensor_constant0
        _tensor_constant1 = self._tensor_constant1
        cat = torch.ops.aten.cat.default([_tensor_constant0, _tensor_constant1], 1);  _tensor_constant0 = _tensor_constant1 = None
        return (cat,)
        
def load_args(reader):
    load_args._version = 0

mod = Repro()

if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)

from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()
_tensor_constant0 = None  # de81e66cd0bc69d023f8cd626a5bc47b62272e2520932a97ea07a10f62b41e8e
_tensor_constant1 = None  # 225ca2a22d58eb631ddb7a654d6c63dbaa2221df52d158edba638dca60837383


cpp_fused_cat_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(8L); i0+=static_cast<long>(1L))
        {
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(384L); i1+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(i1 + (384L*i0)));
                tmp0.store(out_ptr0 + static_cast<long>(i1 + (75648L*i0)));
            }
        }
    }
    #pragma omp parallel num_threads(112)
    {
        {
            #pragma omp for 
            for(long i0=static_cast<long>(0L); i0<static_cast<long>(8L); i0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i1=static_cast<long>(0L); i1<static_cast<long>(192L); i1+=static_cast<long>(16L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(384L); i2+=static_cast<long>(16L))
                    {
                        float tmp0[16*16] __attribute__ ((aligned (16)));
                        at::vec::transpose_mxn<float,16,16>(in_ptr1 + static_cast<long>(i1 + (196L*i2) + (75264L*i0)), static_cast<long>(196L), tmp0, 16);
                        for (long i1_inner = 0; i1_inner < 16; i1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(16L*i1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(i2 + (384L*i1) + (384L*i1_inner) + (75648L*i0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long i1=static_cast<long>(192L); i1<static_cast<long>(196L); i1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long i2=static_cast<long>(0L); i2<static_cast<long>(384L); i2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr1[static_cast<long>(i1 + (196L*i2) + (75264L*i0))];
                        out_ptr1[static_cast<long>(i2 + (384L*i1) + (75648L*i0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    buf2 = empty_strided((8, 197, 384), (75648, 384, 1), device='cpu', dtype=torch.float32)
    buf0 = reinterpret_tensor(buf2, (8, 1, 384), (75648, 384, 1), 0)  # alias
    buf1 = reinterpret_tensor(buf2, (8, 196, 384), (75648, 384, 1), 384)  # alias
    cpp_fused_cat_0(c_void_p(_tensor_constant0.data_ptr()), c_void_p(_tensor_constant1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _tensor_constant0
    _tensor_constant0 = rand_strided((8, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    global _tensor_constant1
    _tensor_constant1 = rand_strided((8, 196, 384), (75264, 1, 196), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)