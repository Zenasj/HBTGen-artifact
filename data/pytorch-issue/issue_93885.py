import torch.nn as nn

isolate_fails_code_str = None


import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

import torch._dynamo.config
import torch._inductor.config
torch._dynamo.config.load_config(b'\x80\x04\x95\x91\x07\x00\x00\x00\x00\x00\x00}\x94(\x8c\x08__name__\x94\x8c\x14torch._dynamo.config\x94\x8c\x07__doc__\x94N\x8c\x0b__package__\x94\x8c\rtorch._dynamo\x94\x8c\n__loader__\x94\x8c\x1a_frozen_importlib_external\x94\x8c\x10SourceFileLoader\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94h\x02\x8c\x04path\x94\x8c?/usr/local/lib/python3.10/site-packages/torch/_dynamo/config.py\x94ub\x8c\x08__spec__\x94\x8c\x11_frozen_importlib\x94\x8c\nModuleSpec\x94\x93\x94)\x81\x94}\x94(h\x0ch\x02\x8c\x06loader\x94h\n\x8c\x06origin\x94h\x0e\x8c\x0cloader_state\x94N\x8c\x1asubmodule_search_locations\x94N\x8c\r_set_fileattr\x94\x88\x8c\x07_cached\x94\x8cX/usr/local/lib/python3.10/site-packages/torch/_dynamo/__pycache__/config.cpython-310.pyc\x94\x8c\r_initializing\x94\x89ub\x8c\x08__file__\x94h\x0e\x8c\n__cached__\x94h\x1b\x8c\x07abspath\x94\x8c\tposixpath\x94h\x1f\x93\x94\x8c\x07dirname\x94h h"\x93\x94\x8c\tlog_level\x94K\x1e\x8c\x0boutput_code\x94\x89\x8c\rlog_file_name\x94N\x8c\x07verbose\x94\x89\x8c\x11output_graph_code\x94\x89\x8c\x12verify_correctness\x94\x89\x8c\x12minimum_call_count\x94K\x01\x8c\x15dead_code_elimination\x94\x88\x8c\x10cache_size_limit\x94K@\x8c\x14specialize_int_float\x94\x88\x8c\x0edynamic_shapes\x94\x89\x8c\x10guard_nn_modules\x94\x89\x8c\x0cnormalize_ir\x94\x89\x8c\x1btraceable_tensor_subclasses\x94\x8f\x94\x8c\x0fsuppress_errors\x94\x89\x8c\x15replay_record_enabled\x94\x89\x8c rewrite_assert_with_torch_assert\x94\x88\x8c\x12print_graph_breaks\x94\x89\x8c\x07disable\x94\x89\x8c*allowed_functions_module_string_ignorelist\x94\x8f\x94(\x8c\x0ctorch._prims\x94\x8c\rtorch._decomp\x94\x8c\x13torch.distributions\x94\x8c\x0btorch._refs\x94\x8c\rtorch.testing\x94\x90\x8c\x0frepro_tolerance\x94G?PbM\xd2\xf1\xa9\xfc\x8c\x16capture_scalar_outputs\x94\x89\x8c\x19enforce_cond_guards_match\x94\x88\x8c\x0coptimize_ddp\x94\x88\x8c\x1araise_on_ctx_manager_usage\x94\x88\x8c\x1craise_on_unsafe_aot_autograd\x94\x89\x8c\rdynamo_import\x94\x8c\rtorch._dynamo\x94\x8c\x0finductor_import\x94\x8c\x0ftorch._inductor\x94\x8c\x18error_on_nested_fx_trace\x94\x88\x8c\tallow_rnn\x94\x89\x8c\x08base_dir\x94\x8c\'/usr/local/lib/python3.10/site-packages\x94\x8c\x0edebug_dir_root\x94\x8c\x19/bsrt/torch_compile_debug\x94\x8c)DO_NOT_USE_legacy_non_fake_example_inputs\x94\x89\x8c\x15_AccessLimitingConfig\x94}\x94(\x8c\n__module__\x94h\x02\x8c\x0b__setattr__\x94h\x02\x8c!_AccessLimitingConfig.__setattr__\x94\x93\x94h\x03Nu\x8c\x15_allowed_config_names\x94\x8f\x94(\x8c\x03sys\x94hP\x8c\x02os\x94h+h.hEhJ\x8c\x12constant_functions\x94h@h*h4h0h1\x8c\x07logging\x94\x8c\x0brepro_after\x94hMh?hAh\x06h\x04h\x01\x8c\x05torch\x94\x8c\x0eexternal_utils\x94h)hKh%h"h&h8h7h6h\x1fh3\x8c!skipfiles_inline_module_allowlist\x94h\x1eh\x03\x8c\nModuleType\x94h/h(h-h\'hGh\x0fhBh$hCh\x1dh,hD\x8c\x0brepro_level\x94hOhIh5\x8c\x0c__builtins__\x94\x90\x8c\x1cget_config_serialization_fns\x94\x8c\x1atorch._dynamo.config_utils\x94hc\x93\x94u.')
torch._inductor.config.load_config(b'\x80\x04\x95\x0f\t\x00\x00\x00\x00\x00\x00}\x94(\x8c\x08__name__\x94\x8c\x16torch._inductor.config\x94\x8c\x07__doc__\x94N\x8c\x0b__package__\x94\x8c\x0ftorch._inductor\x94\x8c\n__loader__\x94\x8c\x1a_frozen_importlib_external\x94\x8c\x10SourceFileLoader\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94h\x02\x8c\x04path\x94\x8cA/usr/local/lib/python3.10/site-packages/torch/_inductor/config.py\x94ub\x8c\x08__spec__\x94\x8c\x11_frozen_importlib\x94\x8c\nModuleSpec\x94\x93\x94)\x81\x94}\x94(h\x0ch\x02\x8c\x06loader\x94h\n\x8c\x06origin\x94h\x0e\x8c\x0cloader_state\x94N\x8c\x1asubmodule_search_locations\x94N\x8c\r_set_fileattr\x94\x88\x8c\x07_cached\x94\x8cZ/usr/local/lib/python3.10/site-packages/torch/_inductor/__pycache__/config.cpython-310.pyc\x94\x8c\r_initializing\x94\x89ub\x8c\x08__file__\x94h\x0e\x8c\n__cached__\x94h\x1b\x8c\x05debug\x94\x89\x8c\x10disable_progress\x94\x88\x8c\x10verbose_progress\x94\x89\x8c\x0bcpp_wrapper\x94\x89\x8c\x03dce\x94\x89\x8c\x14static_weight_shapes\x94\x88\x8c\x0csize_asserts\x94\x88\x8c\x10pick_loop_orders\x94\x88\x8c\x0finplace_buffers\x94\x88\x8c\x11benchmark_harness\x94\x88\x8c\x0fepilogue_fusion\x94\x89\x8c\x15epilogue_fusion_first\x94\x89\x8c\x0fpattern_matcher\x94\x88\x8c\nreordering\x94\x89\x8c\x0cmax_autotune\x94\x89\x8c\x17realize_reads_threshold\x94K\x04\x8c\x17realize_bytes_threshold\x94M\xd0\x07\x8c\x1brealize_acc_reads_threshold\x94K\x08\x8c\x0ffallback_random\x94\x89\x8c\x12implicit_fallbacks\x94\x88\x8c\rprefuse_nodes\x94\x88\x8c\x0btune_layout\x94\x89\x8c\x11aggressive_fusion\x94\x89\x8c\x0fmax_fusion_size\x94K@\x8c\x1bunroll_reductions_threshold\x94K\x08\x8c\x0ecomment_origin\x94\x89\x8c\tis_fbcode\x94h\x02h9\x93\x94\x8c\x0fcompile_threads\x94K \x8c\x13kernel_name_max_ops\x94K\n\x8c\x0finductor_import\x94\x8c\x0ftorch._inductor\x94\x8c\rshape_padding\x94\x89\x8c\x0epermute_fusion\x94\x89\x8c\x1aprofiler_mark_wrapper_call\x94\x89\x8c\x03cpp\x94}\x94(\x8c\n__module__\x94h\x02\x8c\x07threads\x94J\xff\xff\xff\xff\x8c\x0fdynamic_threads\x94\x89\x8c\x07simdlen\x94N\x8c\x0emin_chunk_size\x94M\x00\x10\x8c\x03cxx\x94N\x8c\x03g++\x94\x86\x94\x8c\x15enable_kernel_profile\x94\x89h\x03Nu\x8c\x06triton\x94}\x94(hDh\x02\x8c\ncudagraphs\x94\x88\x8c\x10debug_sync_graph\x94\x89\x8c\x11debug_sync_kernel\x94\x89\x8c\x0bconvolution\x94\x8c\x04aten\x94\x8c\x0edense_indexing\x94\x89\x8c\tmax_tiles\x94K\x02\x8c\x12autotune_pointwise\x94\x88\x8c tiling_prevents_pointwise_fusion\x94\x88\x8c tiling_prevents_reduction_fusion\x94\x88\x8c\x14ordered_kernel_names\x94\x89\x8c\x18descriptive_kernel_names\x94\x89h\x03Nu\x8c\x05trace\x94}\x94(hDh\x02\x8c\x07enabled\x94\x89\x8c\tdebug_log\x94\x88\x8c\x08info_log\x94\x89\x8c\x08fx_graph\x94\x88\x8c\x14fx_graph_transformed\x94\x88\x8c\rir_pre_fusion\x94\x88\x8c\x0eir_post_fusion\x94\x88\x8c\x0boutput_code\x94\x88\x8c\rgraph_diagram\x94\x89\x8c\x0fcompile_profile\x94\x89\x8c\nupload_tar\x94Nh\x03Nu\x8c\x15InductorConfigContext\x94}\x94(hDh\x02\x8c\x0f__annotations__\x94}\x94(\x8c\rstatic_memory\x94\x8c\x08builtins\x94\x8c\x04bool\x94\x93\x94\x8c\x0ematmul_padding\x94hoh-ho\x8c\x12triton_convolution\x94hm\x8c\x03str\x94\x93\x94\x8c\x17rematerialize_threshold\x94hm\x8c\x03int\x94\x93\x94\x8c\x1brematerialize_acc_threshold\x94hvu\x8c\x05_save\x94h\x02\x8c\x1bInductorConfigContext._save\x94\x93\x94\x8c\x06_apply\x94h\x02\x8c\x1cInductorConfigContext._apply\x94\x93\x94\x8c\x08__init__\x94h\x02\x8c\x1eInductorConfigContext.__init__\x94\x93\x94\x8c\t__enter__\x94h\x02\x8c\x1fInductorConfigContext.__enter__\x94\x93\x94\x8c\x08__exit__\x94h\x02\x8c\x1eInductorConfigContext.__exit__\x94\x93\x94h\x03Nu\x8c\x1cget_config_serialization_fns\x94\x8c\x1atorch._dynamo.config_utils\x94h\x87\x93\x94u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.0a0+gitc4ccf7e
# torch cuda version: 11.8
# torch git version: c4ccf7e12147671fdc3535a222260d687c2128a2


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Wed_Sep_21_10:33:58_PDT_2022 
# Cuda compilation tools, release 11.8, V11.8.89 
# Build cuda_11.8.r11.8/compiler.31833905_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', torch.randn([], dtype=torch.float32).cuda())
        self.register_buffer('_tensor_constant1', torch.randn([], dtype=torch.float32).cuda())
        self.register_buffer('_tensor_constant2', torch.randn([], dtype=torch.float32).cuda())



    def forward(self, primals_6, view_1, select, select_1, pow_2, pow_4, bmm, div_2, view_7, permute_8, permute_13, permute_14, permute_15, permute_16, permute_19, tangents_1):
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        maximum = torch.ops.aten.maximum.default(pow_2, lift_fresh_copy);  lift_fresh_copy = None
        expand = torch.ops.aten.expand.default(maximum, [400, 6, 49, 10]);  maximum = None
        div = torch.ops.aten.div.Tensor(select, expand)
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        maximum_1 = torch.ops.aten.maximum.default(pow_4, lift_fresh_copy_1);  lift_fresh_copy_1 = None
        expand_1 = torch.ops.aten.expand.default(maximum_1, [400, 6, 49, 10]);  maximum_1 = None
        div_1 = torch.ops.aten.div.Tensor(select_1, expand_1)
        view_4 = torch.ops.aten.view.default(bmm, [400, 6, 49, 49]);  bmm = None
        _tensor_constant2 = self._tensor_constant2
        lift_fresh_copy_2 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        minimum = torch.ops.aten.minimum.default(primals_6, lift_fresh_copy_2);  lift_fresh_copy_2 = None
        exp = torch.ops.aten.exp.default(minimum);  minimum = None
        full = torch.ops.aten.full.default([4, 64, 64, 60], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(full, tangents_1, 3, 0, 9223372036854775807);  full = tangents_1 = None
        full_1 = torch.ops.aten.full.default([4, 64, 70, 60], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(full_1, slice_scatter_1, 2, 0, 64);  full_1 = slice_scatter_1 = None
        full_2 = torch.ops.aten.full.default([4, 70, 70, 60], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_3 = torch.ops.aten.slice_scatter.default(full_2, slice_scatter_2, 1, 0, 64);  slice_scatter_2 = None
        slice_scatter_4 = torch.ops.aten.slice_scatter.default(full_2, slice_scatter_3, 0, 0, 9223372036854775807);  full_2 = slice_scatter_3 = None
        view_10 = torch.ops.aten.view.default(slice_scatter_4, [4, 10, 7, 10, 7, 60]);  slice_scatter_4 = None
        permute_7 = torch.ops.aten.permute.default(view_10, [0, 1, 3, 2, 4, 5]);  view_10 = None
        clone_8 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_8, [400, 49, 60]);  clone_8 = None
        view_11 = torch.ops.aten.view.default(_unsafe_view_6, [19600, 60]);  _unsafe_view_6 = None
        mm = torch.ops.aten.mm.default(view_11, permute_8);  permute_8 = None
        permute_9 = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_9, view_7);  permute_9 = view_7 = None
        permute_10 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(view_11, [0], True);  view_11 = None
        view_12 = torch.ops.aten.view.default(sum_4, [60]);  sum_4 = None
        permute_11 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        view_13 = torch.ops.aten.view.default(mm, [400, 49, 60]);  mm = None
        view_14 = torch.ops.aten.view.default(view_13, [400, 49, 6, 10]);  view_13 = None
        permute_12 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_9 = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        _unsafe_view_7 = torch.ops.aten._unsafe_view.default(clone_9, [2400, 49, 10]);  clone_9 = None
        bmm_2 = torch.ops.aten.bmm.default(permute_13, _unsafe_view_7);  permute_13 = None
        bmm_3 = torch.ops.aten.bmm.default(_unsafe_view_7, permute_14);  _unsafe_view_7 = permute_14 = None
        view_15 = torch.ops.aten.view.default(bmm_2, [400, 6, 49, 10]);  bmm_2 = None
        view_16 = torch.ops.aten.view.default(bmm_3, [400, 6, 49, 49]);  bmm_3 = None
        mul_1 = torch.ops.aten.mul.Tensor(view_16, div_2);  view_16 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(mul_1, [-1], True)
        mul_2 = torch.ops.aten.mul.Tensor(div_2, sum_5);  div_2 = sum_5 = None
        sub_1 = torch.ops.aten.sub.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(sub_1, [0], True)
        mul_3 = torch.ops.aten.mul.Tensor(sub_1, view_4);  view_4 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_1, exp);  sub_1 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_3, [0, 2, 3], True);  mul_3 = None
        view_17 = torch.ops.aten.view.default(sum_7, [6, 1, 1]);  sum_7 = None
        mul_5 = torch.ops.aten.mul.Tensor(view_17, exp);  view_17 = exp = None
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        le = torch.ops.aten.le.Scalar(primals_6, 4.605170185988092);  primals_6 = None
        where = torch.ops.aten.where.self(le, mul_5, scalar_tensor);  le = mul_5 = None
        view_18 = torch.ops.aten.view.default(mul_4, [2400, 49, 49]);  mul_4 = None
        bmm_4 = torch.ops.aten.bmm.default(permute_15, view_18);  permute_15 = None
        bmm_5 = torch.ops.aten.bmm.default(view_18, permute_16);  view_18 = permute_16 = None
        view_19 = torch.ops.aten.view.default(bmm_4, [400, 6, 10, 49]);  bmm_4 = None
        view_20 = torch.ops.aten.view.default(bmm_5, [400, 6, 49, 10]);  bmm_5 = None
        permute_17 = torch.ops.aten.permute.default(view_19, [0, 1, 3, 2]);  view_19 = None
        neg = torch.ops.aten.neg.default(permute_17)
        div_4 = torch.ops.aten.div.Tensor(div_1, expand_1);  div_1 = None
        mul_6 = torch.ops.aten.mul.Tensor(neg, div_4);  neg = div_4 = None
        div_5 = torch.ops.aten.div.Tensor(permute_17, expand_1);  permute_17 = expand_1 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(mul_6, [3], True);  mul_6 = None
        ge = torch.ops.aten.ge.Scalar(pow_4, 1e-12)
        where_1 = torch.ops.aten.where.self(ge, sum_8, scalar_tensor);  ge = sum_8 = None
        div_6 = torch.ops.aten.div.Tensor(select_1, pow_4);  select_1 = None
        eq = torch.ops.aten.eq.Scalar(pow_4, 0);  pow_4 = None
        where_2 = torch.ops.aten.where.self(eq, scalar_tensor, div_6);  eq = div_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        add_1 = torch.ops.aten.add.Tensor(div_5, mul_7);  div_5 = mul_7 = None
        neg_1 = torch.ops.aten.neg.default(view_20)
        div_8 = torch.ops.aten.div.Tensor(div, expand);  div = None
        mul_8 = torch.ops.aten.mul.Tensor(neg_1, div_8);  neg_1 = div_8 = None
        div_9 = torch.ops.aten.div.Tensor(view_20, expand);  view_20 = expand = None
        sum_9 = torch.ops.aten.sum.dim_IntList(mul_8, [3], True);  mul_8 = None
        ge_1 = torch.ops.aten.ge.Scalar(pow_2, 1e-12)
        where_3 = torch.ops.aten.where.self(ge_1, sum_9, scalar_tensor);  ge_1 = sum_9 = None
        div_10 = torch.ops.aten.div.Tensor(select, pow_2);  select = None
        eq_1 = torch.ops.aten.eq.Scalar(pow_2, 0);  pow_2 = None
        where_4 = torch.ops.aten.where.self(eq_1, scalar_tensor, div_10);  eq_1 = scalar_tensor = div_10 = None
        mul_9 = torch.ops.aten.mul.Tensor(where_3, where_4);  where_3 = where_4 = None
        add_2 = torch.ops.aten.add.Tensor(div_9, mul_9);  div_9 = mul_9 = None
        full_4 = torch.ops.aten.full.default([3, 400, 6, 49, 10], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter = torch.ops.aten.select_scatter.default(full_4, view_15, 0, 2);  view_15 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(full_4, add_1, 0, 1);  add_1 = None
        add_3 = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(full_4, add_2, 0, 0);  full_4 = add_2 = None
        add_4 = torch.ops.aten.add.Tensor(add_3, select_scatter_2);  add_3 = select_scatter_2 = None
        permute_18 = torch.ops.aten.permute.default(add_4, [1, 3, 0, 2, 4]);  add_4 = None
        clone_10 = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        _unsafe_view_8 = torch.ops.aten._unsafe_view.default(clone_10, [400, 49, 180]);  clone_10 = None
        view_21 = torch.ops.aten.view.default(_unsafe_view_8, [19600, 180]);  _unsafe_view_8 = None
        mm_2 = torch.ops.aten.mm.default(view_21, permute_19);  permute_19 = None
        permute_20 = torch.ops.aten.permute.default(view_21, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_20, view_1);  permute_20 = view_1 = None
        permute_21 = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(view_21, [0], True);  view_21 = None
        view_22 = torch.ops.aten.view.default(sum_10, [180]);  sum_10 = None
        permute_22 = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        view_23 = torch.ops.aten.view.default(mm_2, [400, 49, 60]);  mm_2 = None
        slice_9 = torch.ops.aten.slice.Tensor(view_22, 0, 60, 120)
        clone_11 = torch.ops.aten.clone.default(slice_9, memory_format = torch.contiguous_format);  slice_9 = None
        full_like_1 = torch.ops.aten.full_like.default(clone_11, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False, memory_format = torch.preserve_format);  clone_11 = None
        slice_scatter_5 = torch.ops.aten.slice_scatter.default(view_22, full_like_1, 0, 60, 120);  view_22 = full_like_1 = None
        view_24 = torch.ops.aten.view.default(view_23, [4, 10, 10, 7, 7, 60]);  view_23 = None
        permute_23 = torch.ops.aten.permute.default(view_24, [0, 1, 3, 2, 4, 5]);  view_24 = None
        clone_12 = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        _unsafe_view_9 = torch.ops.aten._unsafe_view.default(clone_12, [4, 70, 70, 60]);  clone_12 = None
        return [permute_22, permute_11, sum_6, slice_scatter_5, view_12, where, _unsafe_view_9]

args = [((6, 1, 1), (1, 1, 1), torch.float32, 'cuda'), ((19600, 60), (60, 1), torch.float32, 'cuda'), ((400, 6, 49, 10), (8820, 10, 180, 1), torch.float32, 'cuda'), ((400, 6, 49, 10), (8820, 10, 180, 1), torch.float32, 'cuda'), ((400, 6, 49, 1), (294, 49, 1, 1), torch.float32, 'cuda'), ((400, 6, 49, 1), (294, 49, 1, 1), torch.float32, 'cuda'), ((2400, 49, 49), (2401, 49, 1), torch.float32, 'cuda'), ((400, 6, 49, 49), (14406, 2401, 49, 1), torch.float32, 'cuda'), ((19600, 60), (60, 1), torch.float32, 'cuda'), ((60, 60), (60, 1), torch.float32, 'cuda'), ((2400, 49, 49), (2401, 1, 49), torch.float32, 'cuda'), ((2400, 10, 49), (490, 1, 10), torch.float32, 'cuda'), ((2400, 10, 49), (490, 1, 10), torch.float32, 'cuda'), ((2400, 49, 10), (490, 1, 49), torch.float32, 'cuda'), ((180, 60), (60, 1), torch.float32, 'cuda'), ((4, 64, 64, 60), (245760, 3840, 60, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)


from functools import partial
from torch._dynamo.debug_utils import (
    isolate_fails,
    dump_compiler_graph_state,
)
from functorch.compile import minifier

env_variables = {"CUDA_VISIBLE_DEVICES": "0"}

minifier(
    mod,
    args,
    module_fails=partial(isolate_fails, env=env_variables, compiler_name="inductor", patch_code=isolate_fails_code_str),
    dump_state=partial(dump_compiler_graph_state, compiler_name="inductor"),
)