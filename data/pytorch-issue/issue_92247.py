import torch.nn as nn

import torch
from torch._inductor import config
config.debug = True
torch._dynamo.config.verbose=True

class MockModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        out = torch.cat([x, y], dim=1)
        out+=z
        return out

mod = MockModule().eval()
inputs = (
                torch.randn([1, 64, 16, 16]),
                torch.randn([1, 64, 16, 16]),
                torch.randn([1, 128, 16, 16]),
            )
ref = mod(*inputs)

with torch.no_grad():
    opt_model = torch._dynamo.optimize('inductor')(mod)
    out = opt_model(*inputs)
    out = opt_model(*inputs)
    out = opt_model(*inputs)
print(torch.equal(ref, out))

from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


kernel_cpp_0 = async_compile.cpp('''
#include "/tmp/torchinductor_xiaobing/77/c7773nj5pwikpmm2pwa62rcudlf7p3if7eyqb5k4sjsvewwje4le.h"
extern "C" void kernel(const float* __restrict__ in_ptr0,
                       const float* __restrict__ in_ptr1,
                       const float* __restrict__ in_ptr2,
                       const float* __restrict__ in_ptr3,
                       float* __restrict__ out_ptr0,
                       float* __restrict__ out_ptr1,
                       float* __restrict__ out_ptr2)
{
    {
        for(long i0=0; i0<1024; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + 16*i0);
            tmp0.store(out_ptr0 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16384; i0<16384; i0+=1)
        {
            auto tmp0 = in_ptr0[i0];
            out_ptr0[i0] = tmp0;
        }
    }
    {
        for(long i0=0; i0<1024; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + 16*i0);
            tmp0.store(out_ptr1 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=16384; i0<16384; i0+=1)
        {
            auto tmp0 = in_ptr1[i0];
            out_ptr1[i0] = tmp0;
        }
    }
    {
        for(long i0=0; i0<2048; i0+=1)
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + 16*i0);
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + 16*i0);
            auto tmp2 = tmp0 + tmp1;
            tmp2.store(out_ptr2 + 16*i0);
        }
        #pragma omp simd simdlen(8) 
        for(long i0=32768; i0<32768; i0+=1)
        {
            auto tmp0 = in_ptr2[i0];
            auto tmp1 = in_ptr3[i0];
            auto tmp2 = tmp0 + tmp1;
            out_ptr2[i0] = tmp2;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    buf3 = empty_strided((1, 128, 16, 16), (32768, 256, 16, 1), device='cpu', dtype=torch.float32)
    buf0 = as_strided(buf3, (1, 64, 16, 16), (32768, 256, 16, 1))  # alias
    buf1 = as_strided(buf3, (1, 64, 16, 16), (32768, 256, 16, 1), 16384)  # alias
    buf2 = empty_strided((1, 128, 16, 16), (32768, 256, 16, 1), device='cpu', dtype=torch.float32)
    kernel_cpp_0(c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    del arg1_1
    del arg2_1
    return (buf3, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 64, 16, 16), (16384, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 64, 16, 16), (16384, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((1, 128, 16, 16), (32768, 256, 16, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1, arg1_1, arg2_1]))

buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep(name='buf0', index=c0, size=(16384,))]
buf0.unmet_dependencies = []
buf0.met_dependencies = [MemoryDep(name='arg0_1', index=c0, size=(16384,))]
buf0.group.device = cpu
buf0.group.iteration = ((16384,), ())
buf0.sizes = ([16384], [])
buf0.aliases = ['buf3']
class buf0_loop_body:
    var_ranges = {z0: 16384}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, load, None)
        return store


buf1: SchedulerNode(ComputedBuffer)
buf1.writes = [MemoryDep(name='buf1', index=c0, size=(16384,))]
buf1.unmet_dependencies = []
buf1.met_dependencies = [MemoryDep(name='arg1_1', index=c0, size=(16384,))]
buf1.group.device = cpu
buf1.group.iteration = ((16384,), ())
buf1.sizes = ([16384], [])
buf1.aliases = ['buf3']
class buf1_loop_body:
    var_ranges = {z0: 16384}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg1_1', get_index)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1', get_index_1, load, None)
        return store

buf2: NopKernelSchedulerNode(ConcatKernel)
buf2.writes = [StarDep(name='buf2')]
buf2.unmet_dependencies = [StarDep(name='buf0'), StarDep(name='buf1')]
buf2.met_dependencies = []


buf3: SchedulerNode(ComputedBuffer)
buf3.writes = [MemoryDep(name='buf3', index=c0, size=(32768,))]
buf3.unmet_dependencies = [MemoryDep(name='buf2', index=c0, size=(32768,))]
buf3.met_dependencies = [MemoryDep(name='arg2_1', index=c0, size=(32768,))]
buf3.group.device = cpu
buf3.group.iteration = ((32768,), ())
buf3.sizes = ([32768], [])
class buf3_loop_body:
    var_ranges = {z0: 32768}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf2', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg2_1', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf3', get_index_2, add, None)
        return store

buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep(name='buf0', index=c0, size=(16384,))]
buf0.unmet_dependencies = []
buf0.met_dependencies = [MemoryDep(name='arg0_1', index=c0, size=(16384,))]
buf0.group.device = cpu
buf0.group.iteration = ((16384,), ())
buf0.sizes = ([16384], [])
buf0.aliases = ['buf3']
class buf0_loop_body:
    var_ranges = {z0: 16384}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, load, None)
        return store


buf1: SchedulerNode(ComputedBuffer)
buf1.writes = [MemoryDep(name='buf1', index=c0, size=(16384,))]
buf1.unmet_dependencies = []
buf1.met_dependencies = [MemoryDep(name='arg1_1', index=c0, size=(16384,))]
buf1.group.device = cpu
buf1.group.iteration = ((16384,), ())
buf1.sizes = ([16384], [])
buf1.aliases = ['buf3']
class buf1_loop_body:
    var_ranges = {z0: 16384}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg1_1', get_index)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1', get_index_1, load, None)
        return store


buf2: NopKernelSchedulerNode(ConcatKernel)
buf2.writes = [StarDep(name='buf2')]
buf2.unmet_dependencies = [StarDep(name='buf0'), StarDep(name='buf1')]
buf2.met_dependencies = []


buf3: SchedulerNode(ComputedBuffer)
buf3.writes = [MemoryDep(name='buf3', index=c0, size=(32768,))]
buf3.unmet_dependencies = [MemoryDep(name='buf2', index=c0, size=(32768,))]
buf3.met_dependencies = [MemoryDep(name='arg2_1', index=c0, size=(32768,))]
buf3.group.device = cpu
buf3.group.iteration = ((32768,), ())
buf3.sizes = ([32768], [])
class buf3_loop_body:
    var_ranges = {z0: 32768}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf2', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg2_1', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf3', get_index_2, add, None)
        return store

buf0: SchedulerNode(ComputedBuffer)
buf0.writes = [MemoryDep(name='buf0', index=c0, size=(16384,))]
buf0.unmet_dependencies = []
buf0.met_dependencies = [MemoryDep(name='arg0_1', index=c0, size=(16384,))]
buf0.group.device = cpu
buf0.group.iteration = ((16384,), ())
buf0.sizes = ([16384], [])
buf0.aliases = ['buf2']
class buf0_loop_body:
    var_ranges = {z0: 16384}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, load, None)
        return store


buf1: SchedulerNode(ComputedBuffer)
buf1.writes = [MemoryDep(name='buf1', index=c0, size=(16384,))]
buf1.unmet_dependencies = []
buf1.met_dependencies = [MemoryDep(name='arg1_1', index=c0, size=(16384,))]
buf1.group.device = cpu
buf1.group.iteration = ((16384,), ())
buf1.sizes = ([16384], [])
buf1.aliases = ['buf2']
class buf1_loop_body:
    var_ranges = {z0: 16384}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg1_1', get_index)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1', get_index_1, load, None)
        return store

buf2: NopKernelSchedulerNode(ConcatKernel)
buf2.writes = [StarDep(name='buf2')]
buf2.unmet_dependencies = [StarDep(name='buf0'), StarDep(name='buf1')]
buf2.met_dependencies = []


buf3: SchedulerNode(ComputedBuffer)
buf3.writes = [MemoryDep(name='buf3', index=c0, size=(32768,))]
buf3.unmet_dependencies = [MemoryDep(name='buf2', index=c0, size=(32768,)), StarDep(name='buf2')]
buf3.met_dependencies = [MemoryDep(name='arg2_1', index=c0, size=(32768,))]
buf3.group.device = cpu
buf3.group.iteration = ((32768,), ())
buf3.sizes = ([32768], [])
buf3.mutations = ['buf2']
class buf3_loop_body:
    var_ranges = {z0: 32768}
    index0 = z0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf2', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg2_1', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf3', get_index_2, add, None)
        return store