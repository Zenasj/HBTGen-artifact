import torch.nn as nn

3
# AOT ID: ['0_forward']
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
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
# kernel path: /tmp/torchinductor_yifu/w6/cw6wlc7j53ywtkqjsa2epovtl6wlcpohtogokwqivp2cad6mfbnz.py
# Source Nodes: [prim_redistribute_4], Original ATen: [aten.cat]
# prim_redistribute_4 => cat_1
triton_poi_fused_cat_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
@triton_heuristics.pointwise(
    size_hints=[536870912], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=3, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=132), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'b0714af1b5ca55d285c0aa74c7669268efe45fdd9b7a0a70183b8d89ac8fcac4', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': False, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 536870912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 8388608)
    x0 = xindex % 8388608
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (67108864*x1)), tmp4, other=0.0).to(tl.float32)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 16, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-528482304) + x0 + (67108864*x1)), tmp11, other=0.0).to(tl.float32)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 24, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-1056964608) + x0 + (67108864*x1)), tmp18, other=0.0).to(tl.float32)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1], 32, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tmp22 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-1585446912) + x0 + (67108864*x1)), tmp25, other=0.0).to(tl.float32)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = tmp0 >= tmp23
    tmp30 = tl.full([1], 40, tl.int64)
    tmp31 = tmp0 < tmp30
    tmp32 = tmp29 & tmp31
    tmp33 = tl.load(in_ptr0 + ((-2113929216) + x0 + (67108864*x1)), tmp32, other=0.0).to(tl.float32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tmp0 >= tmp30
    tmp37 = tl.full([1], 48, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp36 & tmp38
    tmp40 = tl.load(in_ptr0 + ((-2642411520) + x0 + (67108864*x1)), tmp39, other=0.0).to(tl.float32)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tmp0 >= tmp37
    tmp44 = tl.full([1], 56, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tl.load(in_ptr0 + ((-3170893824) + x0 + (67108864*x1)), tmp46, other=0.0).to(tl.float32)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp0 >= tmp44
    tmp51 = tl.full([1], 64, tl.int64)
    tmp52 = tmp0 < tmp51
    tmp53 = tl.load(in_ptr0 + ((-3699376128) + x0 + (67108864*x1)), tmp50, other=0.0).to(tl.float32)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp50, tmp53, tmp54)
    tmp56 = tl.where(tmp46, tmp49, tmp55)
    tmp57 = tl.where(tmp39, tmp42, tmp56)
    tmp58 = tl.where(tmp32, tmp35, tmp57)
    tmp59 = tl.where(tmp25, tmp28, tmp58)
    tmp60 = tl.where(tmp18, tmp21, tmp59)
    tmp61 = tl.where(tmp11, tmp14, tmp60)
    tmp62 = tl.where(tmp4, tmp7, tmp61)
    tl.store(out_ptr0 + (x2), tmp62, None)
''', device_str='cuda')
async_compile.wait(globals())
del async_compile
def benchmark_compiled_module(times=10, repeat=10):
    with torch.cuda._DeviceGuard(3):
        torch.cuda.set_device(3)
        # buf38.shape
        # [rank0]:(Pdb) [rank0]:torch.Size([65536, 8192])
        # buf38.stride()
        # [rank0]:(Pdb) [rank0]:(8192, 1)
        # buf39.shape
        # [rank0]:(Pdb) [rank0]:torch.Size([64, 1024, 8192])
        # buf39.stride()
        # [rank0]:(Pdb) [rank0]:(8388608, 8192, 1)
        buf38 = empty_strided_cuda((65536, 8192), (8192, 1), torch.bfloat16)
        buf39 = empty_strided_cuda((64, 1024, 8192), (8388608, 8192, 1), torch.bfloat16)
        stream3 = get_raw_stream(3)
        triton_poi_fused_cat_6.run(buf38, buf39, 536870912, grid=grid(536870912), stream=stream3)
if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
    benchmark_compiled_module()

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.automatic_dynamic_shapes = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True
isolate_fails_code_str = None
# torch version: 2.4.0a0+git71530c8
# torch cuda version: 12.1
# torch git version: 71530c872861d8648aa9b4e797de484d997259db
# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Mon_Apr__3_17:16:06_PDT_2023 
# Cuda compilation tools, release 12.1, V12.1.105 
# Build cuda_12.1.r12.1/compiler.32688072_0 
# GPU Hardware Info: 
# NVIDIA H100 : 8 
from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11):
        wait_tensor = torch.ops._c10d_functional.wait_tensor.default(primals_10);  primals_10 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(wait_tensor, torch.float32)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, rsqrt);  convert_element_type = rsqrt = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_1, primals_1);  convert_element_type_1 = None
        all_gather_into_tensor = torch.ops._c10d_functional.all_gather_into_tensor.default(mul_1, 8, '0');  mul_1 = None
        wait_tensor_1 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        split = torch.ops.aten.split.Tensor(wait_tensor_1, 8);  wait_tensor_1 = None
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6]
        getitem_7 = split[7];  split = None
        cat = torch.ops.aten.cat.default([getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7], 1);  getitem = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = None
        permute = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        view_1 = torch.ops.aten.view.default(cat, [65536, 8192]);  cat = None
        mm = torch.ops.aten.mm.default(view_1, permute)
        view_2 = torch.ops.aten.view.default(mm, [8, 8192, 1024]);  mm = None
        permute_1 = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        mm_1 = torch.ops.aten.mm.default(view_1, permute_1)
        view_5 = torch.ops.aten.view.default(mm_1, [8, 8192, 128]);  mm_1 = None
        permute_2 = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        mm_2 = torch.ops.aten.mm.default(view_1, permute_2);  view_1 = None
        view_8 = torch.ops.aten.view.default(mm_2, [8, 8192, 128]);  mm_2 = None
        view_10 = torch.ops.aten.view.default(view_2, [8, 8192, 8, 128]);  view_2 = None
        view_11 = torch.ops.aten.view.default(view_5, [8, 8192, 1, 128]);  view_5 = None
        view_12 = torch.ops.aten.view.default(view_8, [8, 8192, 1, 128]);  view_8 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        view_13 = torch.ops.aten.view.default(convert_element_type_8, [8, 8192, 8, -1, 2]);  convert_element_type_8 = None
        view_as_complex = torch.ops.aten.view_as_complex.default(view_13);  view_13 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        view_14 = torch.ops.aten.view.default(convert_element_type_9, [8, 8192, 1, -1, 2]);  convert_element_type_9 = None
        view_as_complex_1 = torch.ops.aten.view_as_complex.default(view_14);  view_14 = None
        slice_1 = torch.ops.aten.slice.Tensor(primals_11, 0, 0, 8192);  primals_11 = None
        view_15 = torch.ops.aten.view.default(slice_1, [1, 8192, 1, 64]);  slice_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_as_complex, view_15);  view_as_complex = None
        view_as_real = torch.ops.aten.view_as_real.default(mul_2);  mul_2 = None
        view_16 = torch.ops.aten.view.default(view_as_real, [8, 8192, 8, 128]);  view_as_real = None
        mul_3 = torch.ops.aten.mul.Tensor(view_as_complex_1, view_15);  view_as_complex_1 = None
        view_as_real_1 = torch.ops.aten.view_as_real.default(mul_3);  mul_3 = None
        view_17 = torch.ops.aten.view.default(view_as_real_1, [8, 8192, 1, 128]);  view_as_real_1 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(view_16, torch.bfloat16);  view_16 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(view_17, torch.bfloat16);  view_17 = None
        slice_2 = torch.ops.aten.slice.Tensor(convert_element_type_11, 0, 0, 9223372036854775807);  convert_element_type_11 = None
        slice_3 = torch.ops.aten.slice.Tensor(slice_2, 1, 0, 9223372036854775807);  slice_2 = None
        slice_4 = torch.ops.aten.slice.Tensor(slice_3, 2, 0, 9223372036854775807);  slice_3 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(slice_4, 3);  slice_4 = None
        slice_5 = torch.ops.aten.slice.Tensor(unsqueeze, 4, 0, 9223372036854775807);  unsqueeze = None
        expand = torch.ops.aten.expand.default(slice_5, [8, 8192, 1, 8, 128]);  slice_5 = None
        view_18 = torch.ops.aten.view.default(expand, [8, 8192, 8, 128]);  expand = None
        slice_6 = torch.ops.aten.slice.Tensor(view_12, 0, 0, 9223372036854775807);  view_12 = None
        slice_7 = torch.ops.aten.slice.Tensor(slice_6, 1, 0, 9223372036854775807);  slice_6 = None
        slice_8 = torch.ops.aten.slice.Tensor(slice_7, 2, 0, 9223372036854775807);  slice_7 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(slice_8, 3);  slice_8 = None
        slice_9 = torch.ops.aten.slice.Tensor(unsqueeze_1, 4, 0, 9223372036854775807);  unsqueeze_1 = None
        expand_1 = torch.ops.aten.expand.default(slice_9, [8, 8192, 1, 8, 128]);  slice_9 = None
        view_19 = torch.ops.aten.view.default(expand_1, [8, 8192, 8, 128]);  expand_1 = None
        permute_3 = torch.ops.aten.permute.default(convert_element_type_10, [0, 2, 1, 3]);  convert_element_type_10 = None
        permute_4 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        permute_5 = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3]);  view_19 = None
        run_and_save_rng_state = torch.ops.higher_order.run_and_save_rng_state(torch.ops.aten._scaled_dot_product_flash_attention.default, permute_3, permute_4, permute_5, 0.0, True, scale = 0.08838834764831843);  permute_3 = permute_4 = permute_5 = None
        getitem_41 = run_and_save_rng_state[0]
        getitem_42 = run_and_save_rng_state[1];  run_and_save_rng_state = None
        getitem_8 = getitem_42[0];  getitem_42 = None
        permute_6 = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
        view_20 = torch.ops.aten.view.default(permute_6, [8, 8192, -1]);  permute_6 = None
        permute_7 = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        view_22 = torch.ops.aten.view.default(view_20, [65536, 1024]);  view_20 = None
        mm_3 = torch.ops.aten.mm.default(view_22, permute_7);  view_22 = None
        view_23 = torch.ops.aten.view.default(mm_3, [8, 8192, 8192]);  mm_3 = None
        split_1 = torch.ops.aten.split.Tensor(view_23, 1024, 1);  view_23 = None
        getitem_17 = split_1[0]
        getitem_18 = split_1[1]
        getitem_19 = split_1[2]
        getitem_20 = split_1[3]
        getitem_21 = split_1[4]
        getitem_22 = split_1[5]
        getitem_23 = split_1[6]
        getitem_24 = split_1[7];  split_1 = None
        cat_1 = torch.ops.aten.cat.default([getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24]);  getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = None
        reduce_scatter_tensor = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_1, 'sum', 8, '0');  cat_1 = None
        wait_tensor_2 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        add_1 = torch.ops.aten.add.Tensor(wait_tensor, wait_tensor_2);  wait_tensor_2 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_14, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_2 = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(convert_element_type_14, rsqrt_1);  convert_element_type_14 = rsqrt_1 = None
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(mul_4, torch.bfloat16);  mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_15, primals_6);  convert_element_type_15 = None
        all_gather_into_tensor_1 = torch.ops._c10d_functional.all_gather_into_tensor.default(mul_5, 8, '0');  mul_5 = None
        wait_tensor_4 = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        split_2 = torch.ops.aten.split.Tensor(wait_tensor_4, 8);  wait_tensor_4 = None
        getitem_25 = split_2[0]
        getitem_26 = split_2[1]
        getitem_27 = split_2[2]
        getitem_28 = split_2[3]
        getitem_29 = split_2[4]
        getitem_30 = split_2[5]
        getitem_31 = split_2[6]
        getitem_32 = split_2[7];  split_2 = None
        cat_2 = torch.ops.aten.cat.default([getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32], 1);  getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = None
        permute_8 = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        view_26 = torch.ops.aten.view.default(cat_2, [65536, 8192]);  cat_2 = None
        mm_4 = torch.ops.aten.mm.default(view_26, permute_8)
        view_27 = torch.ops.aten.view.default(mm_4, [8, 8192, 3584]);  mm_4 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_18)
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_18, sigmoid);  convert_element_type_18 = sigmoid = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        permute_9 = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        mm_5 = torch.ops.aten.mm.default(view_26, permute_9);  view_26 = None
        view_30 = torch.ops.aten.view.default(mm_5, [8, 8192, 3584]);  mm_5 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_19, view_30);  convert_element_type_19 = view_30 = None
        permute_10 = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        view_33 = torch.ops.aten.view.default(mul_7, [65536, 3584]);  mul_7 = None
        mm_6 = torch.ops.aten.mm.default(view_33, permute_10);  view_33 = None
        view_34 = torch.ops.aten.view.default(mm_6, [8, 8192, 8192]);  mm_6 = None
        split_3 = torch.ops.aten.split.Tensor(view_34, 1024, 1);  view_34 = None
        getitem_33 = split_3[0]
        getitem_34 = split_3[1]
        getitem_35 = split_3[2]
        getitem_36 = split_3[3]
        getitem_37 = split_3[4]
        getitem_38 = split_3[5]
        getitem_39 = split_3[6]
        getitem_40 = split_3[7];  split_3 = None
        cat_3 = torch.ops.aten.cat.default([getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40]);  getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = None
        reduce_scatter_tensor_1 = torch.ops._c10d_functional.reduce_scatter_tensor.default(cat_3, 'sum', 8, '0');  cat_3 = None
        wait_tensor_5 = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        add_3 = torch.ops.aten.add.Tensor(add_1, wait_tensor_5);  add_1 = wait_tensor_5 = None
        permute_15 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        return [add_3, primals_1, primals_6, wait_tensor, permute, permute_1, permute_2, view_15, permute_7, permute_8, permute_9, permute_15, getitem_41]
        
def load_args(reader):
    buf0 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (8192,), dtype=torch.bfloat16, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (1024, 8192), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (128, 8192), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (128, 8192), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (8192, 1024), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 16384, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf5, (8192,), dtype=torch.bfloat16, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf6, (3584, 8192), dtype=torch.bfloat16, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf7, (3584, 8192), dtype=torch.bfloat16, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 58720256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf8, (8192, 3584), dtype=torch.bfloat16, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 134217728, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf9, (8, 1024, 8192), dtype=torch.bfloat16, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 8388608, device=device(type='cuda', index=0), dtype_hint=torch.complex64)
    reader.tensor(buf10, (16384, 64), dtype=torch.complex64, is_leaf=True)  # primals_11
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch.testing._internal.distributed.fake_pg import FakeStore
    torch.distributed.init_process_group("fake", store=FakeStore(), rank=0, world_size=8)
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)