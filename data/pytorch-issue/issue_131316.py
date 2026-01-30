import torch.nn as nn

import torch
import torch.nn.functional as F
from torch import nn

class ModelArgs:
    dim: int = 512
    n_heads: int = 8
    n_kv_heads: int = 8
    max_seq_len: int = 128

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bs, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bs, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seqlen, self.n_kv_heads, self.head_dim)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(output)

def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    args = ModelArgs()
    model = Attention(args).to(device)
    model = torch.compile(model)

    x = torch.randn(1, args.max_seq_len, args.dim, requires_grad=True, device=device)
    x = x.bfloat16()

    output = model(x)
    loss = output.sum()
    loss.backward()

    print("Input gradient shape:", x.grad.shape)
    print("Input gradient sum:", x.grad.sum().item())
    print("Output shape:", output.shape)
    print("Output sum:", output.sum().item())

if __name__ == "__main__":
    main()

# AOT ID: ['0_backward']
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
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_larsenra/hk/chkkhpdnd4hzy5dv7x545hlv6sypde24g6vkdrk6sesiu6ce5l4q.py
# Source Nodes: [], Original ATen: [aten.add]

triton_poi_fused_add_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='hip', index=0, cc='gfx90a', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, multi_processor_count=None), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': None, 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'is_hip': True},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x0), tmp4, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


async_compile.wait(globals())
del async_compile

def call(args):
    view, permute_3, permute_4, permute_5, getitem, getitem_1, getitem_6, getitem_7, permute_10, permute_18, permute_22, permute_26, tangents_1 = args
    args.clear()
    assert_size_stride(view, (128, 512), (512, 1))
    assert_size_stride(permute_3, (1, 8, 128, 64), (65536, 64, 512, 1))
    assert_size_stride(permute_4, (1, 8, 128, 64), (65536, 64, 512, 1))
    assert_size_stride(permute_5, (1, 8, 128, 64), (65536, 64, 512, 1))
    assert_size_stride(getitem, (1, 8, 128, 64), (65536, 64, 512, 1))
    assert_size_stride(getitem_1, (1, 8, 128), (1024, 128, 1))
    assert_size_stride(getitem_6, (), ())
    assert_size_stride(getitem_7, (), ())
    assert_size_stride(permute_10, (512, 512), (512, 1))
    assert_size_stride(permute_18, (512, 512), (512, 1))
    assert_size_stride(permute_22, (512, 512), (512, 1))
    assert_size_stride(permute_26, (512, 512), (512, 1))
    assert_size_stride(tangents_1, (1, 128, 512), (65536, 512, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (512, 128), (1, 512), 0), reinterpret_tensor(getitem, (128, 512), (512, 1), 0), out=buf0)
        buf1 = empty_strided_cuda((128, 512), (512, 1), torch.bfloat16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(tangents_1, (128, 512), (512, 1), 0), permute_10, out=buf1)
        del permute_10
        del tangents_1
        # Source Nodes: [], Original ATen: [aten._scaled_dot_product_flash_attention_backward]
        buf2 = aten._scaled_dot_product_flash_attention_backward.default(reinterpret_tensor(buf1, (1, 8, 128, 64), (65536, 64, 512, 1), 0), permute_3, permute_4, permute_5, getitem, getitem_1, None, None, 128, 128, 0.0, True, getitem_6, getitem_7, scale=0.125)
        del getitem
        del getitem_1
        del getitem_6
        del getitem_7
        del permute_3
        del permute_4
        del permute_5
        buf3 = buf2[0]
        buf4 = buf2[1]
        buf5 = buf2[2]
        del buf2
        buf6 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (512, 128), (1, 512), 0), view, out=buf6)
        buf7 = buf1; del buf1  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf5, (128, 512), (512, 1), 0), permute_18, out=buf7)
        del permute_18
        buf8 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (512, 128), (1, 512), 0), view, out=buf8)
        buf9 = reinterpret_tensor(buf5, (128, 512), (512, 1), 0); del buf5  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (128, 512), (512, 1), 0), permute_22, out=buf9)
        del permute_22
        buf10 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (512, 128), (1, 512), 0), view, out=buf10)
        del view
        buf11 = reinterpret_tensor(buf4, (128, 512), (512, 1), 0); del buf4  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (128, 512), (512, 1), 0), permute_26, out=buf11)
        del buf3
        del permute_26
        buf12 = reinterpret_tensor(buf11, (1, 128, 512), (65536, 512, 1), 0); del buf11  # reuse
        # Source Nodes: [], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_0.run(buf12, buf7, buf9, 65536, grid=grid(65536), stream=stream0)
        del buf7
        del buf9
    return (reinterpret_tensor(buf10, (512, 512), (512, 1), 0), reinterpret_tensor(buf8, (512, 512), (512, 1), 0), reinterpret_tensor(buf6, (512, 512), (512, 1), 0), reinterpret_tensor(buf0, (512, 512), (512, 1), 0), buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    view = rand_strided((128, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_3 = rand_strided((1, 8, 128, 64), (65536, 64, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_4 = rand_strided((1, 8, 128, 64), (65536, 64, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_5 = rand_strided((1, 8, 128, 64), (65536, 64, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((1, 8, 128, 64), (65536, 64, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((1, 8, 128), (1024, 128, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    permute_10 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_18 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_22 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_26 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((1, 128, 512), (65536, 512, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([view, permute_3, permute_4, permute_5, getitem, getitem_1, getitem_6, getitem_7, permute_10, permute_18, permute_22, permute_26, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)