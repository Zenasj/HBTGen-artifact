import torch.nn as nn

py
import torch

torch.manual_seed(420)

x = torch.randn(1, 3, 2, 2)

class Model(torch.nn.Module):
    def forward(self, x):
        out = x
        mask1 = out > 0
        out[mask1] *= 0.5
        return out

func = Model().to('cpu')

res1 = func(x)
print(res1)
# tensor([[[[-1.6977,  0.3187],
#           [ 0.0391, -0.4140]],
# 
#          [[ 0.7586,  0.0236],
#           [ 0.4217, -0.2261]],
# 
#          [[ 0.0173, -0.3422],
#           [ 0.3414, -0.8155]]]])

jit_func = torch.compile(func)
res2 = jit_func(x)
print(res2)
# tensor([0.1594, 0.0195, 0.3793, 0.0118, 0.2109, 0.0086, 0.1707])

import torch._inductor.overrides

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\r\x00\x00\x00log_file_nameq\x01NX\x07\x00\x00\x00verboseq\x02\x89X\x12\x00\x00\x00verify_correctnessq\x03\x89X\x12\x00\x00\x00minimum_call_countq\x04K\x01X\x15\x00\x00\x00dead_code_eliminationq\x05\x88X\x10\x00\x00\x00cache_size_limitq\x06K@X\x0e\x00\x00\x00specialize_intq\x07\x89X\x0e\x00\x00\x00dynamic_shapesq\x08\x89X\x18\x00\x00\x00assume_static_by_defaultq\t\x88X\x18\x00\x00\x00automatic_dynamic_shapesq\n\x88X\x19\x00\x00\x00allow_ignore_mark_dynamicq\x0b\x89X\x10\x00\x00\x00guard_nn_modulesq\x0c\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\rc__builtin__\nset\nq\x0e]q\x0f\x85q\x10Rq\x11X\x0f\x00\x00\x00suppress_errorsq\x12\x89X\x15\x00\x00\x00replay_record_enabledq\x13\x88X \x00\x00\x00rewrite_assert_with_torch_assertq\x14\x88X\x12\x00\x00\x00print_graph_breaksq\x15\x89X\x15\x00\x00\x00print_specializationsq\x16\x89X\x19\x00\x00\x00summarize_dim_constraintsq\x17\x89X\x07\x00\x00\x00disableq\x18\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x19h\x0e]q\x1a(X\r\x00\x00\x00torch._decompq\x1bX\r\x00\x00\x00torch.testingq\x1cX\x0c\x00\x00\x00torch._primsq\x1dX\x13\x00\x00\x00torch.distributionsq\x1eX\x0b\x00\x00\x00torch._refsq\x1fe\x85q Rq!X\x12\x00\x00\x00repro_forward_onlyq"\x89X\x0f\x00\x00\x00repro_toleranceq#G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq$\x89X \x00\x00\x00capture_dynamic_output_shape_opsq%\x89X\x19\x00\x00\x00enforce_cond_guards_matchq&\x88X\x0c\x00\x00\x00optimize_ddpq\'\x88X\x10\x00\x00\x00skip_fsdp_guardsq(\x88X\x19\x00\x00\x00skip_nnmodule_hook_guardsq)\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq*\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq+\x89X\x17\x00\x00\x00raise_on_backend_changeq,\x89X\x18\x00\x00\x00error_on_nested_fx_traceq-\x88X\t\x00\x00\x00allow_rnnq.\x89X\x12\x00\x00\x00error_on_recompileq/\x89X\x08\x00\x00\x00base_dirq0X\x12\x00\x00\x00/workspace/pytorchq1X\x12\x00\x00\x00DEBUG_DIR_VAR_NAMEq2X\x17\x00\x00\x00TORCH_COMPILE_DEBUG_DIRq3X\x0e\x00\x00\x00debug_dir_rootq4X&\x00\x00\x00/workspace/pytorch/torch_compile_debugq5X\x13\x00\x00\x00_save_config_ignoreq6h\x0e]q7(X\x12\x00\x00\x00constant_functionsq8X\x0b\x00\x00\x00repro_afterq9X\x0b\x00\x00\x00repro_levelq:X!\x00\x00\x00skipfiles_inline_module_allowlistq;e\x85q<Rq=u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x88X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x16\x00\x00\x00max_autotune_pointwiseq\x10\x89X\x11\x00\x00\x00max_autotune_gemmq\x11\x89X\x15\x00\x00\x00search_autotune_cacheq\x12\x89X\x13\x00\x00\x00autotune_in_subprocq\x13\x89X\x19\x00\x00\x00coordinate_descent_tuningq\x14\x89X\x17\x00\x00\x00realize_reads_thresholdq\x15K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x16M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x17K\x08X\x0f\x00\x00\x00fallback_randomq\x18\x89X\x12\x00\x00\x00implicit_fallbacksq\x19\x88X\x11\x00\x00\x00aggressive_fusionq\x1a\x89X\x0f\x00\x00\x00max_fusion_sizeq\x1bK@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x1cK\x08X\x0e\x00\x00\x00comment_originq\x1d\x89X\x0e\x00\x00\x00conv_1x1_as_mmq\x1e\x89X\x10\x00\x00\x00split_reductionsq\x1f\x88X\x0e\x00\x00\x00lowmem_dropoutq \x88X\x10\x00\x00\x00benchmark_kernelq!\x89X\x14\x00\x00\x00is_nightly_or_sourceq"\x88X\x12\x00\x00\x00developer_warningsq#\x88X\x0f\x00\x00\x00compile_threadsq$K X\x10\x00\x00\x00global_cache_dirq%NX\x13\x00\x00\x00kernel_name_max_opsq&K\nX\r\x00\x00\x00shape_paddingq\'\x89X\x0e\x00\x00\x00permute_fusionq(\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq)\x89X\x18\x00\x00\x00_raise_error_for_testingq*\x89X\x0c\x00\x00\x00_profile_varq+X\x00\x00\x00\x00q,X\x11\x00\x00\x00profile_bandwidthq-\x89X\x17\x00\x00\x00profile_bandwidth_regexq.h,X\x13\x00\x00\x00disable_cpp_codegenq/\x89X\x0b\x00\x00\x00cpp.threadsq0J\xff\xff\xff\xffX\x16\x00\x00\x00cpp.no_redundant_loopsq1\x88X\x13\x00\x00\x00cpp.dynamic_threadsq2\x89X\x0b\x00\x00\x00cpp.simdlenq3NX\x12\x00\x00\x00cpp.min_chunk_sizeq4M\x00\x10X\x07\x00\x00\x00cpp.cxxq5NX\x03\x00\x00\x00g++q6\x86q7X\x19\x00\x00\x00cpp.enable_kernel_profileq8\x89X\x12\x00\x00\x00cpp.weight_prepackq9\x88X\x11\x00\x00\x00triton.cudagraphsq:\x89X\x16\x00\x00\x00triton.cudagraph_treesq;\x89X"\x00\x00\x00triton.slow_path_cudagraph_assertsq<\x88X"\x00\x00\x00triton.fast_path_cudagraph_assertsq=\x89X\x1c\x00\x00\x00triton.skip_cudagraph_warmupq>\x89X\x17\x00\x00\x00triton.debug_sync_graphq?\x89X\x18\x00\x00\x00triton.debug_sync_kernelq@\x89X\x15\x00\x00\x00triton.dense_indexingqA\x89X\x10\x00\x00\x00triton.max_tilesqBK\x02X\x19\x00\x00\x00triton.autotune_pointwiseqC\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionqD\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionqE\x88X\x1f\x00\x00\x00triton.assert_indirect_indexingqF\x88X\x1a\x00\x00\x00triton.unique_kernel_namesqG\x89X\x18\x00\x00\x00triton.descriptive_namesqHX\r\x00\x00\x00original_atenqIX\x1c\x00\x00\x00triton.persistent_reductionsqJ\x88X\x16\x00\x00\x00triton.divisible_by_16qK\x88X\x10\x00\x00\x00triton.max_blockqL}qM(X\x01\x00\x00\x00XqNM\x00\x08X\x01\x00\x00\x00YqOM\x00\x04X\x01\x00\x00\x00ZqPM\x00\x04uX\x12\x00\x00\x00triton.store_cubinqQ\x89X\x16\x00\x00\x00triton.spill_thresholdqRK\x00X\r\x00\x00\x00trace.enabledqS\x88X\x0f\x00\x00\x00trace.debug_logqT\x89X\x0e\x00\x00\x00trace.info_logqU\x89X\x0e\x00\x00\x00trace.fx_graphqV\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqW\x88X\x13\x00\x00\x00trace.ir_pre_fusionqX\x88X\x14\x00\x00\x00trace.ir_post_fusionqY\x88X\x11\x00\x00\x00trace.output_codeqZ\x88X\x13\x00\x00\x00trace.graph_diagramq[\x89X\x15\x00\x00\x00trace.compile_profileq\\\x89X\x13\x00\x00\x00_save_config_ignoreq]c__builtin__\nset\nq^]q_X\x10\x00\x00\x00trace.upload_tarq`a\x85qaRqbu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x15\x00\x00\x00functionalize_rng_opsq\x01\x89X\x16\x00\x00\x00fake_tensor_allow_metaq\x02\x88X\x0c\x00\x00\x00debug_assertq\x03\x89X\x14\x00\x00\x00static_weight_shapesq\x04\x88X\x03\x00\x00\x00cseq\x05\x88X\x10\x00\x00\x00max_dist_from_bwq\x06K\x03X\x11\x00\x00\x00debug_partitionerq\x07\x88u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.1.0a0+git7beac10
# torch cuda version: None
# torch git version: 7beac103eecf3828e627ecc22cb289948cafac04


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        mul = torch.ops.aten.mul.Tensor(arg2_1, 0.5)
        index_put = torch.ops.aten.index_put.default(arg0_1, [arg1_1], mul);  arg1_1 = None
        copy_ = torch.ops.aten.copy_.default(arg0_1, index_put);  arg0_1 = None
        copy__1 = torch.ops.aten.copy_.default(arg2_1, mul);  arg2_1 = mul = None
        return (index_put,)
        
args = []
args.append(rand_strided((1, 3, 2, 2), (12, 4, 2, 1), torch.float32, 'cpu'))  # shape (1, 3, 2, 2), stride (12, 4, 2, 1)
args.append(rand_strided((1, 3, 2, 2), (12, 4, 2, 1), torch.bool, 'cpu'))  # shape (1, 3, 2, 2), stride (12, 4, 2, 1)
args.append(rand_strided((7,), (1,), torch.float32, 'cpu'))  # shape (7,), stride (1,)
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)