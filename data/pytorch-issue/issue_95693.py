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
import torch._functorch.config
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x18\x00\x00\x00assume_static_by_defaultq\x0b\x89X\x10\x00\x00\x00guard_nn_modulesq\x0c\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\rc__builtin__\nset\nq\x0e]q\x0f\x85q\x10Rq\x11X\x0f\x00\x00\x00suppress_errorsq\x12\x89X\x15\x00\x00\x00replay_record_enabledq\x13\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x14\x88X\x12\x00\x00\x00print_graph_breaksq\x15\x89X\x07\x00\x00\x00disableq\x16\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x17h\x0e]q\x18(X\r\x00\x00\x00torch._decompq\x19X\r\x00\x00\x00torch.testingq\x1aX\x0c\x00\x00\x00torch._primsq\x1bX\x0b\x00\x00\x00torch._refsq\x1cX\x13\x00\x00\x00torch.distributionsq\x1de\x85q\x1eRq\x1fX\x12\x00\x00\x00repro_forward_onlyq \x89X\x0f\x00\x00\x00repro_toleranceq!G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq"\x89X \x00\x00\x00capture_dynamic_output_shape_opsq#\x89X\x19\x00\x00\x00enforce_cond_guards_matchq$\x88X\x0c\x00\x00\x00optimize_ddpq%\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq&\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq\'\x89X\x17\x00\x00\x00raise_on_backend_changeq(\x89X\x18\x00\x00\x00error_on_nested_fx_traceq)\x88X\t\x00\x00\x00allow_rnnq*\x89X\x08\x00\x00\x00base_dirq+X@\x00\x00\x00/opt/conda/envs/centerpoint-nuscenes/lib/python3.8/site-packagesq,X\x0e\x00\x00\x00debug_dir_rootq-Xa\x00\x00\x00/home/users/chenrui17/baidu/hac-aiacc/AIAK-MODEL/pytorch/centerpoint-nuscenes/torch_compile_debugq.X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq/\x89X\x13\x00\x00\x00_save_config_ignoreq0h\x0e]q1(X\x12\x00\x00\x00constant_functionsq2X\x0b\x00\x00\x00repro_afterq3X!\x00\x00\x00skipfiles_inline_module_allowlistq4X\x0b\x00\x00\x00repro_levelq5e\x85q6Rq7u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x15\x00\x00\x00search_autotune_cacheq\x10\x88X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x89X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\x0b\x00\x00\x00tune_layoutq\x16\x89X\x11\x00\x00\x00aggressive_fusionq\x17\x89X\x0f\x00\x00\x00max_fusion_sizeq\x18K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x19K\x08X\x0e\x00\x00\x00comment_originq\x1a\x89X\x12\x00\x00\x00developer_warningsq\x1b\x88X\x0f\x00\x00\x00compile_threadsq\x1cK X\x11\x00\x00\x00global_cache_pathq\x1dNX\x13\x00\x00\x00kernel_name_max_opsq\x1eK\nX\r\x00\x00\x00shape_paddingq\x1f\x89X\x0e\x00\x00\x00permute_fusionq \x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq!\x89X\x18\x00\x00\x00_raise_error_for_testingq"\x89X\x0c\x00\x00\x00_profile_varq#X\x00\x00\x00\x00q$X\x11\x00\x00\x00profile_bandwidthq%\x89X\x17\x00\x00\x00profile_bandwidth_regexq&h$X\x0b\x00\x00\x00cpp.threadsq\'J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq(\x89X\x0b\x00\x00\x00cpp.simdlenq)NX\x12\x00\x00\x00cpp.min_chunk_sizeq*M\x00\x10X\x07\x00\x00\x00cpp.cxxq+NX\x03\x00\x00\x00g++q,\x86q-X\x19\x00\x00\x00cpp.enable_kernel_profileq.\x89X\x12\x00\x00\x00cpp.weight_prepackq/\x88X\x11\x00\x00\x00triton.cudagraphsq0\x89X\x17\x00\x00\x00triton.debug_sync_graphq1\x89X\x18\x00\x00\x00triton.debug_sync_kernelq2\x89X\x12\x00\x00\x00triton.convolutionq3X\x04\x00\x00\x00atenq4X\x15\x00\x00\x00triton.dense_indexingq5\x89X\x10\x00\x00\x00triton.max_tilesq6K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq7\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq8\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq9\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq:\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq;\x89X\x1c\x00\x00\x00triton.persistent_reductionsq<\x88X\x10\x00\x00\x00triton.max_blockq=}q>(X\x01\x00\x00\x00Xq?M\x00\x08X\x01\x00\x00\x00Yq@M\x00\x04X\x01\x00\x00\x00ZqAM\x00\x04uX\r\x00\x00\x00trace.enabledqB\x89X\x0f\x00\x00\x00trace.debug_logqC\x88X\x0e\x00\x00\x00trace.info_logqD\x89X\x0e\x00\x00\x00trace.fx_graphqE\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqF\x88X\x13\x00\x00\x00trace.ir_pre_fusionqG\x88X\x14\x00\x00\x00trace.ir_post_fusionqH\x88X\x11\x00\x00\x00trace.output_codeqI\x88X\x13\x00\x00\x00trace.graph_diagramqJ\x89X\x15\x00\x00\x00trace.compile_profileqK\x89X\x10\x00\x00\x00trace.upload_tarqLNu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.0.dev20230227+cu117
# torch cuda version: 11.7
# torch git version: 1e2e6e78c68c8df58d1498bc495629e56d433598


# CUDA Info:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2022 NVIDIA Corporation
# Built on Tue_May__3_18:49:52_PDT_2022
# Cuda compilation tools, release 11.7, V11.7.64
# Build cuda_11.7.r11.7/compiler.31294372_0

# GPU Hardware Info:
# NVIDIA A100-SXM4-40GB : 8


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, primals_3, primals_10, convert_element_type_1, convolution, squeeze_1, relu, convert_element_type_5, unsqueeze_6, tangents_1, tangents_2, tangents_3, tangents_4):
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sum_1 = torch.ops.aten.sum.dim_IntList(tangents_4, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(tangents_4, relu, convert_element_type_5, [1], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  tangents_4 = convert_element_type_5 = None
        getitem_2 = convolution_backward[0]
        getitem_3 = convolution_backward[1];  convolution_backward = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        le = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(le, scalar_tensor, getitem_2);  le = scalar_tensor = getitem_2 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_8, [0, 2, 3])
        sub_1 = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_6);  convert_element_type_2 = unsqueeze_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_8, sub_1)
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_7, [0, 2, 3]);  mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(sum_2, 3.814697265625e-06)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(mul_8, 0);  mul_8 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, 3);  unsqueeze_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(sum_3, 3.814697265625e-06)
        mul_10 = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_11 = torch.ops.aten.mul.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(mul_11, 0);  mul_11 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 2);  unsqueeze_10 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, 3);  unsqueeze_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(squeeze_1, primals_3);  primals_3 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(mul_12, 0);  mul_12 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 2);  unsqueeze_13 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 3);  unsqueeze_14 = None
        mul_13 = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_12);  sub_1 = unsqueeze_12 = None
        sub_3 = torch.ops.aten.sub.Tensor(convert_element_type_8, mul_13);  convert_element_type_8 = mul_13 = None
        sub_4 = torch.ops.aten.sub.Tensor(sub_3, unsqueeze_9);  sub_3 = unsqueeze_9 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_15);  sub_4 = unsqueeze_15 = None
        mul_15 = torch.ops.aten.mul.Tensor(sum_3, squeeze_1);  sum_3 = squeeze_1 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(mul_14, torch.float16);  mul_14 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_10, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_10, primals_10, convert_element_type_1, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_10 = primals_10 = convert_element_type_1 = None
        getitem_5 = convolution_backward_1[0]
        getitem_6 = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(getitem_6, torch.float32);  getitem_6 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(sum_4, torch.float32);  sum_4 = None
        return [convert_element_type_11, convert_element_type_12, mul_15, sum_2, convert_element_type_6, convert_element_type_7, None, None, None, getitem_5]

args = [((64,), (1,), torch.float32, 'cuda'), ((16, 64, 128, 128), (1048576, 1, 8192, 64), torch.float16, 'cuda'), ((64, 64, 3, 3), (576, 9, 3, 1), torch.float16, 'cuda'), ((16, 64, 128, 128), (1048576, 1, 8192, 64), torch.float16, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((16, 64, 128, 128), (1048576, 1, 8192, 64), torch.float16, 'cuda'), ((1, 64, 3, 3), (576, 9, 3, 1), torch.float16, 'cuda'), ((1, 64, 1, 1), (64, 1, 1, 1), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((64,), (1,), torch.float32, 'cuda'), ((), (), torch.int64, 'cuda'), ((16, 1, 128, 128), (16384, 16384, 128, 1), torch.float16, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)


from functools import partial
from torch._dynamo.debug_utils import (
    isolate_fails,
    dump_compiler_graph_state,
)
from functorch.compile import minifier

env_variables = {"CUDA_VISIBLE_DEVICES": "1"}

minifier(
    mod,
    args,
    module_fails=partial(isolate_fails, env=env_variables, compiler_name="inductor", patch_code=isolate_fails_code_str),
    dump_state=partial(dump_compiler_graph_state, compiler_name="inductor"),
)