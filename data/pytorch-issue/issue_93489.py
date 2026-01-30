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
#include "/tmp/torchinductor_xiaobing/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(float* __restrict__ in_out_ptr0,
                       float* __restrict__ in_out_ptr1)
{
    for(long i0=0; i0<384; i0+=1)
    {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + 16*i0);
        tmp0.store(in_out_ptr0 + 16*i0);
    }
    #pragma omp simd simdlen(8)
    for(long i0=6144; i0<6144; i0+=1)
    {
        auto tmp0 = in_out_ptr0[i0];
        in_out_ptr0[i0] = tmp0;
    }
    for(long i0=0; i0<384; i0+=1)
    {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + 16*i0);
        tmp0.store(in_out_ptr1 + 16*i0);
    }
    #pragma omp simd simdlen(8)
    for(long i0=6144; i0<6144; i0+=1)
    {
        auto tmp0 = in_out_ptr1[i0];
        in_out_ptr1[i0] = tmp0;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    buf0 = torch.ops.mkl._mkl_linear(arg9_1, arg2_1, arg0_1, arg1_1, 8)
    del arg0_1
    del arg1_1
    del arg2_1
    buf1 = torch.ops.mkl._mkl_linear(arg9_1, arg5_1, arg3_1, arg4_1, 8)
    del arg3_1
    del arg4_1
    del arg5_1
    del arg9_1
    buf3 = as_strided(buf0, (12, 8, 64), (64, 768, 1)); del buf0  # reuse
    buf4 = as_strided(buf1, (12, 64, 8), (64, 1, 768)); del buf1  # reuse
    kernel_cpp_0(c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    buf5 = empty_strided((12, 8, 8), (64, 8, 1), device='cpu', dtype=torch.float32)
    aten.bmm.out(buf3, buf4, out=buf5)
    return (as_strided(buf5, (1, 12, 8, 8), (768, 64, 8, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((1, 8, 128), (1024, 128, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1]))

import torch._dynamo
import torch.fx.experimental.optimization as optimization
import copy

from typing import Dict, List, Optional

import time
import torch.profiler as profiler
from torch.fx import symbolic_trace
from torch._inductor import config
config.debug = True

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = 64
        self.query = torch.nn.Linear(128, 768)
        self.key = torch.nn.Linear(128, 768)
        self.value = torch.nn.Linear(128, 768)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        #value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        return attention_scores


x = torch.randn(1, 8, 128)

model = Model().eval()
y = model(x)

with torch.no_grad():
    opt_model = torch._dynamo.optimize('inductor')(model)

with torch.no_grad():
    for i in range(2):
        y1 = opt_model(x)

import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    buf0 = empty_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    aten.addmm.out(arg1_1, as_strided(arg4_1, (8, 128), (128, 1)), as_strided(arg0_1, (128, 768), (1, 128)), beta=1, alpha=1, out=buf0)
    del arg0_1
    del arg1_1
    buf1 = empty_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    aten.addmm.out(arg3_1, as_strided(arg4_1, (8, 128), (128, 1)), as_strided(arg2_1, (128, 768), (1, 128)), beta=1, alpha=1, out=buf1)
    del arg2_1
    del arg3_1
    del arg4_1
    buf2 = empty_strided((12, 8, 8), (64, 8, 1), device='cpu', dtype=torch.float32)
    aten.bmm.out(as_strided(buf0, (12, 8, 64), (64, 768, 1)), as_strided(buf1, (12, 64, 8), (64, 1, 768)), out=buf2)
    return (as_strided(buf2, (1, 12, 8, 8), (768, 64, 8, 1)), )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((1, 8, 128), (1024, 128, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]))

from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()
_frozen_param1 = None  # b03a1db7099d8c2033f876544a2843fd102c9dfb8c3dbe30cf6b3caac588a57c
_frozen_param3 = None  # 17a2699e5272226e84f2c584fe6478af996d80fe857e57a269aa3c73ccdf2f27
_frozen_param8 = None  # 81adf4c423d3c390e6fbe96d2cd309504103fdf64017ac524e7e3f813ba3aee3
_frozen_param9 = None  # 337b9aa40a6f93ac98f1f584ceecf8a1f97c2dbb68be865014340706ba530483
_frozen_param10 = None  # a5da9ee8de1ab16dbfb160231c25566234a86d1548e0b20c010546b5432fa4f2
_frozen_param11 = None  # 337b9aa40a6f93ac98f1f584ceecf8a1f97c2dbb68be865014340706ba530483


cpp_fused_addmm_0 = async_compile.cpp('''
#include "/tmp/torchinductor_jianan/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                tmp2.store(in_out_ptr1 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg6_1, = args
    args.clear()
    assert_size_stride(arg6_1, (1, 8, 128), (1024, 128, 1))
    buf0 = torch.ops.mkl._mkl_linear(reinterpret_tensor(arg6_1, (8, 128), (128, 1), 0), _frozen_param9, _frozen_param8, None, 8)
    buf1 = torch.ops.mkl._mkl_linear(reinterpret_tensor(arg6_1, (8, 128), (128, 1), 0), _frozen_param11, _frozen_param10, None, 8)
    del arg6_1
    buf2 = buf0; del buf0  # reuse
    buf3 = buf1; del buf1  # reuse
    cpp_fused_addmm_0(c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(_frozen_param1.data_ptr()), c_void_p(_frozen_param3.data_ptr()))
    buf4 = empty((12, 8, 8), device='cpu', dtype=torch.float32)
    # Source Nodes: [attention_scores], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf2, (12, 8, 64), (64, 768, 1), 0), reinterpret_tensor(buf3, (12, 64, 8), (64, 1, 768), 0), out=buf4)
    return (reinterpret_tensor(buf4, (1, 12, 8, 8), (768, 64, 8, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global _frozen_param1
    _frozen_param1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    global _frozen_param3
    _frozen_param3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    global _frozen_param8
    _frozen_param8 = rand_strided((768, 128), (128, 1), device='cpu', dtype=torch.float32)
    global _frozen_param9
    _frozen_param9 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    global _frozen_param10
    _frozen_param10 = rand_strided((768, 128), (128, 1), device='cpu', dtype=torch.float32)
    global _frozen_param11
    _frozen_param11 = rand_strided((2310369, 1), (1, 0), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((1, 8, 128), (1024, 128, 1), device='cpu', dtype=torch.float32)
    fn = lambda: call([arg6_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)