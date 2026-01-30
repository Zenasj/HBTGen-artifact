import numpy as np

import torch
import torch._inductor.config

from torch._inductor.ir import Pointwise
from torch._inductor.lowering import register_lowering
from torch._inductor.virtualized import ops

torch._inductor.config.triton.debug_sync_kernel = True

test_inductor_ops = torch.library.Library("test_inductor_ops", "DEF")
impl_cuda = torch.library.Library("test_inductor_ops", "IMPL", "CUDA")
impl_meta = torch.library.Library("test_inductor_ops", "IMPL", "Meta")

def register_op():
	test_inductor_ops.define(
		"jagged_to_padded_dense(Tensor input, Tensor offsets, SymInt max_seq_len, Scalar pad_value) -> Tensor"
	)

	def j2pd_meta(inp, offsets, max_seq_len, pad_value):
		return torch.empty(
			(offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
			device=inp.device,
			dtype=inp.dtype,
		)

	def j2pd_cuda(inp, offsets, max_seq_len, pad_value):
		res = torch.full(
			(offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
			pad_value,
			device=inp.device,
			dtype=inp.dtype,
		)
		for b in range(offsets.shape[0] - 1):
			for r in range(offsets[b + 1] - offsets[b]):
				res[b][r] = inp[offsets[b] + r]
		return res

	def j2pd_lowering(inp, offsets, max_seq_len, pad_value):
		offsets_loader = offsets.make_loader()
		inp_loader = inp.make_loader()
		jagged_len = inp.get_size()[0]
		offsets_dtype = offsets.get_dtype()

		def inner_fn(index):
			batch_idx, seq_idx, emb_idx = index

			begin_idx = ops.indirect_indexing(
				offsets_loader([batch_idx]),
				jagged_len + 1,
			)
			end_idx = offsets_loader([batch_idx + 1])
			jagged_idx = begin_idx + seq_idx

			return ops.masked(
				ops.lt(
					ops.index_expr(jagged_idx, offsets_dtype),
					end_idx,
				),
				lambda: inp_loader([jagged_idx, emb_idx]),
				pad_value,
			)

		return Pointwise.create(
			device=inp.get_device(),
			dtype=inp.get_dtype(),
			inner_fn=inner_fn,
			ranges=[offsets.get_size()[0] - 1, max_seq_len, inp.get_size()[1]],
		)

	impl_meta.impl("jagged_to_padded_dense", j2pd_meta)
	impl_cuda.impl("jagged_to_padded_dense", j2pd_cuda)

	register_lowering(
		torch.ops.test_inductor_ops.jagged_to_padded_dense, type_promotion_kind=None
	)(j2pd_lowering)


def assertEqual(x, y):
    assert (x-y).abs().max() == 0

def sanity_check():
	def fn(inp, offsets, max_seq_len):
		return torch.ops.test_inductor_ops.jagged_to_padded_dense(
			inp, offsets, max_seq_len, 60.0
		)

	inp = torch.rand((9, 96), device="cuda")
	offsets = torch.tensor([0, 2, 5, 9], dtype=torch.int32, device="cuda")
	max_seq_len = 4

	res = fn(inp, offsets, max_seq_len)
	assertEqual(inp[0], res[0][0])
	assertEqual(inp[1], res[0][1])
	assertEqual(inp[2], res[1][0])
	assertEqual(inp[3], res[1][1])
	assertEqual(inp[5], res[2][0])
	assertEqual(inp[8], res[2][3])

	fn_opt = torch.compile(fn)

	assertEqual(
		fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
	)

def zero_failure():
	def fn(inp, offsets, max_seq_len):
		inp = torch.bmm(inp, torch.ones((1, 96, 1), device="cuda")).view((0, 1))
		return torch.ops.test_inductor_ops.jagged_to_padded_dense(
			inp, offsets, max_seq_len, 60.0
		)

	inp = torch.rand((1, 0, 96), device="cuda")
	offsets = torch.zeros(1025, device="cuda", dtype=torch.int32)
	max_seq_len = 20

	fn_opt = torch.compile(fn)

	assertEqual(
		fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
	)

register_op()
sanity_check()
torch.cuda.synchronize()

zero_failure()
torch.cuda.synchronize()

from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_dberard/jr/cjrujf6uggqreawjgfneo5cpvb56krxxdkh5lc3krqqnimkjf5zf.py
# Source Nodes: [ones], Original ATen: [aten.ones]
# ones => full_default
triton_poi_fused_ones_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_dberard/nn/cnnz5aqhqadbqgypy65ava4ystmid6rur64cssrn2mef2oq6qr7c.py
# Source Nodes: [jagged_to_padded_dense], Original ATen: [test_inductor_ops.jagged_to_padded_dense]
# jagged_to_padded_dense => jagged_to_padded_dense
triton_poi_fused_jagged_to_padded_dense_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*i32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 20)
    x0 = xindex % 20
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + x1), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (0))  # <-- BAD LINE
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tl.device_assert((0 <= tmp0) & (tmp0 < 1), "index out of bounds: 0 <= tmp0 < 1")
    tmp2 = tmp0 + x0
    tmp3 = tmp2 < tmp1
    tmp6 = tl.where(tmp3, tmp5, 60.0)
    tl.store(out_ptr0 + (x2), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg1_1, (1025, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1, 96, 1), (96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [ones], Original ATen: [aten.ones]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_ones_0.run(buf0, 96, grid=grid(96), stream=stream0)
        torch.cuda.synchronize()
        buf1 = empty_strided((1, 0, 1), (0, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [bmm, ones], Original ATen: [aten.bmm, aten.ones]
        extern_kernels.bmm(arg0_1, buf0, out=buf1)
        del arg0_1
        del buf0
        torch.cuda.synchronize()
        buf2 = empty_strided((1024, 20, 1), (20, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [jagged_to_padded_dense], Original ATen: [test_inductor_ops.jagged_to_padded_dense]
        triton_poi_fused_jagged_to_padded_dense_1.run(arg1_1, buf1, buf2, 20480, grid=grid(20480), stream=stream0)
        del arg1_1
        del buf1
        torch.cuda.synchronize()
        return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 0, 96), (96, 96, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1025, ), (1, ), device='cuda:0', dtype=torch.int32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_dberard/jr/cjrujf6uggqreawjgfneo5cpvb56krxxdkh5lc3krqqnimkjf5zf.py
# Source Nodes: [ones], Original ATen: [aten.ones]
# ones => full_default
triton_poi_fused_ones_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[128], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 1.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_dberard/4x/c4xr4ivy5xx6unt7ycaau25b2d5zdw2p74hwxgv4ogdyhvvo4mqp.py
# Source Nodes: [jagged_to_padded_dense], Original ATen: [test_inductor_ops.jagged_to_padded_dense]
# jagged_to_padded_dense => jagged_to_padded_dense
triton_poi_fused_jagged_to_padded_dense_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*i32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 20)
    x0 = xindex % 20
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + x1), None, eviction_policy='evict_last')
    tl.device_assert((0 <= tmp0) & (tmp0 < 3), "index out of bounds: 0 <= tmp0 < 3")
    tmp2 = tmp0 + x0
    tmp3 = tmp2 < tmp1
    tmp4 = tl.load(in_ptr1 + (tmp0 + x0), tmp3, other=0)
    tmp5 = tl.where(tmp3, tmp4, 60.0)
    tl.store(out_ptr0 + (x2), tmp5, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 2, 96), (192, 96, 1))
    assert_size_stride(arg1_1, (1025, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1, 96, 1), (96, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [ones], Original ATen: [aten.ones]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_ones_0.run(buf0, 96, grid=grid(96), stream=stream0)
        torch.cuda.synchronize()
        buf1 = empty_strided((1, 2, 1), (2, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [bmm, ones], Original ATen: [aten.bmm, aten.ones]
        extern_kernels.bmm(arg0_1, buf0, out=buf1)
        del arg0_1
        del buf0
        torch.cuda.synchronize()
        buf2 = empty_strided((1024, 20, 1), (20, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [jagged_to_padded_dense], Original ATen: [test_inductor_ops.jagged_to_padded_dense]
        triton_poi_fused_jagged_to_padded_dense_1.run(arg1_1, buf1, buf2, 20480, grid=grid(20480), stream=stream0)
        del arg1_1
        del buf1
        torch.cuda.synchronize()
        return (buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 2, 96), (192, 96, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1025, ), (1, ), device='cuda:0', dtype=torch.int32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_dberard/eb/cebqzsmggyiz4u765smutzdz5lfmyhbv4sdpxrlbzzeovloz4rpy.py
# Source Nodes: [jagged_to_padded_dense], Original ATen: [test_inductor_ops.jagged_to_padded_dense]
# jagged_to_padded_dense => jagged_to_padded_dense
triton_poi_fused_jagged_to_padded_dense_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*i32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 20)
    x0 = xindex % 20
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + x1), None, eviction_policy='evict_last')
    tl.device_assert((0 <= tmp0) & (tmp0 < 1), "index out of bounds: 0 <= tmp0 < 1")
    tmp2 = tmp0 + x0
    tmp3 = tmp2 < tmp1
    tmp4 = tl.load(in_ptr1 + (tmp0 + x0), tmp3, other=0)
    tmp5 = tl.where(tmp3, tmp4, 60.0)
    tl.store(out_ptr0 + (x2), tmp5, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg1_1, (1025, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1024, 20, 1), (20, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [jagged_to_padded_dense], Original ATen: [test_inductor_ops.jagged_to_padded_dense]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_jagged_to_padded_dense_0.run(arg1_1, arg0_1, buf0, 20480, grid=grid(20480), stream=stream0)
        del arg0_1
        del arg1_1
        torch.cuda.synchronize()
        return (buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((0, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1025, ), (1, ), device='cuda:0', dtype=torch.int32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)