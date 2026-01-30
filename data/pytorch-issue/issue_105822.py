import torch
 
def fn(out, src, index):
  out.scatter_(0, index, src)
  return out
 
out = torch.zeros(8, dtype=torch.int64, device='cuda')
src = torch.tensor([1001, 1002, 1003], dtype=torch.int64, device='cuda')
index = torch.tensor([1, 3, 5], dtype=torch.int64, device='cuda')
 
compiled_f = torch.compile(fn, backend='inductor',
                              options={'trace.enabled':True,
                                       'trace.graph_diagram':True})
 
out = compiled_f(out, src, index)

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
 
 
# kernel path: /tmp/torchinductor_surakav/ve/cve56yx3hkxlcxgcjzkzrwkfqjk32wabeioxkqmj6wzktob6ndln.py
# Original ATen: aten.scatter
 
# aten.scatter => scatter
triton_poi_fused_scatter_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
 
@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')
 
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
 
 
# kernel path: /tmp/torchinductor_surakav/ot/cotufkhkaxlsx7zez6h2ksaw3znejjzmdja62ubpjuiga5xnawzq.py
# Original ATen: aten.scatter
 
# aten.scatter => scatter
triton_poi_fused_scatter_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
 
@pointwise(size_hints=[4], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tl.device_assert(((0 <= tmp0) & (tmp0 < 8)) | ~xmask, "index out of bounds: 0 <= tmp0 < 8")
    tl.store(out_ptr0 + (tl.broadcast_to(tmp0, [XBLOCK])), tmp1, xmask)
''')
 
 
# kernel path: /tmp/torchinductor_surakav/3i/c3igscqoaoc2w6f2rm3zzbumbrsp3453ilgp42nn463kkpl5aed2.py
# Original ATen:
 
triton_poi_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
 
@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')
 
 
async_compile.wait(globals())
del async_compile
 
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, ), (1, ))
    assert_size_stride(arg1_1, (3, ), (1, ))
    assert_size_stride(arg2_1, (3, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, ), (1, ), device='cuda', dtype=torch.int64)
        stream0 = get_cuda_stream(0)
        triton_poi_fused_scatter_0.run(arg0_1, buf0, 8, grid=grid(8), stream=stream0)
        triton_poi_fused_scatter_1.run(arg2_1, arg1_1, buf0, 3, grid=grid(3), stream=stream0)
        del arg1_1
        del arg2_1
        triton_poi_fused_2.run(buf0, arg0_1, 8, grid=grid(8), stream=stream0)
        del arg0_1
        return (buf0, )
 
 
def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg1_1 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg2_1 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1]), times=times, repeat=repeat)
 
 
if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)