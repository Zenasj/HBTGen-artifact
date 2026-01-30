py
import torch
from torch import Tensor
from typing import *


import torch
import triton
import triton.language as tl

@triton.jit
def add_one_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = x + 1
    tl.store(out_ptr + offsets, output, mask=mask)

def add_one(x, out):
    n_elements = x.numel()
    add_one_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

# @torch.library.custom_op("_reinplacing::add_one", mutates_args={"result"})
# def add_one(x: torch.Tensor, result: torch.Tensor) -> None:
#     result.copy_(x + 1)

class MySin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x)
        add_one(x, out)
        ctx.save_for_backward(out)
        return out
 
    @staticmethod
    def backward(ctx, grad):
        saved, = ctx.saved_tensors
        out = torch.empty_like(grad)
        add_one(saved, out)
        return out

@torch.compile
def f(x):
    return MySin.apply(x)

x = torch.randn(3, requires_grad=True, device="cuda")
y = f(x)

py

# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_rzou/kk/ckk4zwgremsdtggt3kdtom7kzh3n65jwybnknzqfcnj7jnds46p6.py
# Topologically Sorted Source Nodes: [autograd_function_apply], Original ATen: []
# Source node to ATen node mapping:
#   autograd_function_apply => triton_kernel_wrapper_functional_proxy
# Graph fragment:
#   %triton_kernel_wrapper_functional_proxy : [num_users=1] = call_function[target=torch.ops.higher_order.triton_kernel_wrapper_functional](args = (), kwargs = {kernel_idx: 0, constant_args_idx: 0, grid: [(3, 1, 1)], kwargs: {in_ptr0: %primals_1, out_ptr: %permute}, tensors_to_clone: [out_ptr]})
triton_poi_fused_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'DBE9DA6C318A5F8C8C0398F5A3ADEA939A6DA06D9FEB6055149E6E52EBE1EF99', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = float("nan")
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# Original path: /home/rzou/dev/debug-cpu1/pt-debug-cpu1/foo.py:10
add_one_kernel_0 = async_compile.triton('add_one_kernel', '''

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.user_autotune(
    configs=[],
    inductor_meta={'kernel_name': 'add_one_kernel_0', 'backend_hash': 'DBE9DA6C318A5F8C8C0398F5A3ADEA939A6DA06D9FEB6055149E6E52EBE1EF99', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {3: 4}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    filename=__file__,
    custom_kernel=True,
)
@triton.jit
def add_one_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = x + 1
    tl.store(out_ptr + offsets, output, mask=mask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, = args
    args.clear()
    assert_size_stride(primals_1, (3, ), (1, ))
    buf0 = empty_strided_cuda((3, ), (1, ), torch.float32)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((3, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [autograd_function_apply], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(buf1, 3, grid=grid(3), stream=stream0)
        # Topologically Sorted Source Nodes: [autograd_function_apply], Original ATen: []
        def grid_wrapper_for_add_one_kernel_0(meta):
            return (3, 1, 1)

        add_one_kernel_0.run(primals_1, buf1, 3, 4, grid=grid_wrapper_for_add_one_kernel_0, stream=stream0)
        del primals_1
    return (buf1, buf0, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)