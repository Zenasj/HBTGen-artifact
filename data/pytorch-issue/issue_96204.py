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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08M\x00\x08X\x0e\x00\x00\x00specialize_intq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x88X\x18\x00\x00\x00assume_static_by_defaultq\x0b\x89X\x10\x00\x00\x00guard_nn_modulesq\x0c\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\rc__builtin__\nset\nq\x0e]q\x0f\x85q\x10Rq\x11X\x0f\x00\x00\x00suppress_errorsq\x12\x89X\x15\x00\x00\x00replay_record_enabledq\x13\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x14\x88X\x12\x00\x00\x00print_graph_breaksq\x15\x89X\x07\x00\x00\x00disableq\x16\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x17h\x0e]q\x18(X\r\x00\x00\x00torch._decompq\x19X\x0b\x00\x00\x00torch._refsq\x1aX\r\x00\x00\x00torch.testingq\x1bX\x13\x00\x00\x00torch.distributionsq\x1cX\x0c\x00\x00\x00torch._primsq\x1de\x85q\x1eRq\x1fX\x12\x00\x00\x00repro_forward_onlyq \x89X\x0f\x00\x00\x00repro_toleranceq!G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq"\x89X \x00\x00\x00capture_dynamic_output_shape_opsq#\x89X\x19\x00\x00\x00enforce_cond_guards_matchq$\x88X\x0c\x00\x00\x00optimize_ddpq%\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq&\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq\'\x89X\x17\x00\x00\x00raise_on_backend_changeq(\x89X\x18\x00\x00\x00error_on_nested_fx_traceq)\x88X\t\x00\x00\x00allow_rnnq*\x89X\x08\x00\x00\x00base_dirq+X#\x00\x00\x00/data/output/t5-hf-compile-dynamic/q,X\x0e\x00\x00\x00debug_dir_rootq-h,X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq.\x89X\x13\x00\x00\x00_save_config_ignoreq/h\x0e]q0(X\x0b\x00\x00\x00repro_afterq1X\x12\x00\x00\x00constant_functionsq2X!\x00\x00\x00skipfiles_inline_module_allowlistq3X\x0b\x00\x00\x00repro_levelq4e\x85q5Rq6u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x15\x00\x00\x00search_autotune_cacheq\x10\x89X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x89X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\x0b\x00\x00\x00tune_layoutq\x16\x89X\x11\x00\x00\x00aggressive_fusionq\x17\x89X\x0f\x00\x00\x00max_fusion_sizeq\x18K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x19K\x08X\x0e\x00\x00\x00comment_originq\x1a\x89X\x10\x00\x00\x00benchmark_kernelq\x1b\x89X\x12\x00\x00\x00developer_warningsq\x1c\x88X\x0f\x00\x00\x00compile_threadsq\x1dK X\x11\x00\x00\x00global_cache_pathq\x1eNX\x13\x00\x00\x00kernel_name_max_opsq\x1fK\nX\r\x00\x00\x00shape_paddingq \x89X\x0e\x00\x00\x00permute_fusionq!\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq"\x89X\x18\x00\x00\x00_raise_error_for_testingq#\x89X\x0c\x00\x00\x00_profile_varq$X\x00\x00\x00\x00q%X\x11\x00\x00\x00profile_bandwidthq&\x89X\x17\x00\x00\x00profile_bandwidth_regexq\'h%X\x0b\x00\x00\x00cpp.threadsq(J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq)\x89X\x0b\x00\x00\x00cpp.simdlenq*NX\x12\x00\x00\x00cpp.min_chunk_sizeq+M\x00\x10X\x07\x00\x00\x00cpp.cxxq,NX\x03\x00\x00\x00g++q-\x86q.X\x19\x00\x00\x00cpp.enable_kernel_profileq/\x89X\x12\x00\x00\x00cpp.weight_prepackq0\x88X\x11\x00\x00\x00triton.cudagraphsq1\x89X\x17\x00\x00\x00triton.debug_sync_graphq2\x89X\x18\x00\x00\x00triton.debug_sync_kernelq3\x89X\x15\x00\x00\x00triton.dense_indexingq4\x89X\x10\x00\x00\x00triton.max_tilesq5K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq6\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq7\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq8\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq9\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq:\x89X\x1c\x00\x00\x00triton.persistent_reductionsq;\x88X\x10\x00\x00\x00triton.max_blockq<}q=(X\x01\x00\x00\x00Xq>M\x00\x08X\x01\x00\x00\x00Yq?M\x00\x04X\x01\x00\x00\x00Zq@M\x00\x04uX\r\x00\x00\x00trace.enabledqA\x89X\x0f\x00\x00\x00trace.debug_logqB\x88X\x0e\x00\x00\x00trace.info_logqC\x89X\x0e\x00\x00\x00trace.fx_graphqD\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqE\x88X\x13\x00\x00\x00trace.ir_pre_fusionqF\x88X\x14\x00\x00\x00trace.ir_post_fusionqG\x88X\x11\x00\x00\x00trace.output_codeqH\x88X\x13\x00\x00\x00trace.graph_diagramqI\x89X\x15\x00\x00\x00trace.compile_profileqJ\x89X\x10\x00\x00\x00trace.upload_tarqKNu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.1.0.dev20230304+cu117
# torch cuda version: 11.7
# torch git version: f1d60f587239a2abdb913619faa61e81ee393ecc


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, arg0_1):
        sym_size = torch.ops.aten.sym_size(arg0_1, 1);  arg0_1 = None
        ge = sym_size >= 150;  sym_size = None
        return (ge,)

args = [((3, 2), (2, 1), torch.int64, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='symbolic')(*args)


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