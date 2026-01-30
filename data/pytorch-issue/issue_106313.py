import torch.nn as nn

import torch
import torch._dynamo
import roma


def myfn(X):
    out = roma.procrustes(X, regularization=0)
    return "OK"

X = torch.rand([1, 8, 8])
print(f"{torch.compile(myfn, backend='eager')(X)=}")
torch._dynamo.reset()
print(f"{torch.compile(myfn, backend='aot_eager')(X)=}")
torch._dynamo.reset()
print(f"{torch.compile(myfn, backend='inductor')(X)=}")
torch._dynamo.reset()

X = X.to("cuda")
print(f"{torch.compile(myfn, backend='eager')(X)=}")
torch._dynamo.reset()
print(f"{torch.compile(myfn, backend='aot_eager')(X)=}")
torch._dynamo.reset()
print(f"{torch.compile(myfn, backend='inductor')(X)=}")

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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\x0c\x00\x00\x00torch._primsq\x18X\r\x00\x00\x00torch.testingq\x19X\x0b\x00\x00\x00torch._refsq\x1aX\x13\x00\x00\x00torch.distributionsq\x1bX\r\x00\x00\x00torch._decompq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x17\x00\x00\x00raise_on_backend_changeq&\x89X\x18\x00\x00\x00error_on_nested_fx_traceq\'\x88X\t\x00\x00\x00allow_rnnq(\x89X\x08\x00\x00\x00base_dirq)XC\x00\x00\x00/nfs/tools/3d/caiman/conda/envs/caiman/lib/python3.10/site-packagesq*X\x0e\x00\x00\x00debug_dir_rootq+X.\x00\x00\x00/home/plekhac/crocoman-toy/torch_compile_debugq,X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq-\x89X\x13\x00\x00\x00_save_config_ignoreq.h\r]q/(X\x0b\x00\x00\x00repro_afterq0X!\x00\x00\x00skipfiles_inline_module_allowlistq1X\x12\x00\x00\x00constant_functionsq2X\x0b\x00\x00\x00repro_levelq3e\x85q4Rq5u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x17\x00\x00\x00realize_reads_thresholdq\x10K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x11M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x12K\x08X\x0f\x00\x00\x00fallback_randomq\x13\x89X\x12\x00\x00\x00implicit_fallbacksq\x14\x88X\x0b\x00\x00\x00tune_layoutq\x15\x89X\x11\x00\x00\x00aggressive_fusionq\x16\x89X\x0f\x00\x00\x00max_fusion_sizeq\x17K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x18K\x08X\x0e\x00\x00\x00comment_originq\x19\x89X\x12\x00\x00\x00developer_warningsq\x1a\x89X\x0f\x00\x00\x00compile_threadsq\x1bK X\x13\x00\x00\x00kernel_name_max_opsq\x1cK\nX\r\x00\x00\x00shape_paddingq\x1d\x89X\x0e\x00\x00\x00permute_fusionq\x1e\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq\x1f\x89X\x18\x00\x00\x00_raise_error_for_testingq \x89X\x0b\x00\x00\x00cpp.threadsq!J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq"\x89X\x0b\x00\x00\x00cpp.simdlenq#NX\x12\x00\x00\x00cpp.min_chunk_sizeq$M\x00\x10X\x07\x00\x00\x00cpp.cxxq%NX\x03\x00\x00\x00g++q&\x86q\'X\x19\x00\x00\x00cpp.enable_kernel_profileq(\x89X\x12\x00\x00\x00cpp.weight_prepackq)\x88X\x11\x00\x00\x00triton.cudagraphsq*\x89X\x17\x00\x00\x00triton.debug_sync_graphq+\x89X\x18\x00\x00\x00triton.debug_sync_kernelq,\x89X\x15\x00\x00\x00triton.dense_indexingq-\x89X\x10\x00\x00\x00triton.max_tilesq.K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq/\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq0\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq1\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq2\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq3\x89X\x1c\x00\x00\x00triton.persistent_reductionsq4\x89X\r\x00\x00\x00trace.enabledq5\x89X\x0f\x00\x00\x00trace.debug_logq6\x88X\x0e\x00\x00\x00trace.info_logq7\x89X\x0e\x00\x00\x00trace.fx_graphq8\x88X\x1a\x00\x00\x00trace.fx_graph_transformedq9\x88X\x13\x00\x00\x00trace.ir_pre_fusionq:\x88X\x14\x00\x00\x00trace.ir_post_fusionq;\x88X\x11\x00\x00\x00trace.output_codeq<\x88X\x13\x00\x00\x00trace.graph_diagramq=\x89X\x15\x00\x00\x00trace.compile_profileq>\x89X\x10\x00\x00\x00trace.upload_tarq?Nu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.1
# torch cuda version: 11.8
# torch git version: e9ebda29d87ce0916ab08c06ab26fd3766a870e5


# CUDA Info: 
# nvcc not found
# GPU Hardware Info: 
# NVIDIA A100 80GB PCIe : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, arg0_1):
        _linalg_svd = torch.ops.aten._linalg_svd.default(arg0_1, True);  arg0_1 = None
        getitem = _linalg_svd[0]
        getitem_1 = _linalg_svd[1]
        getitem_2 = _linalg_svd[2];  _linalg_svd = None
        permute = torch.ops.aten.permute.default(getitem_2, [0, 2, 1]);  getitem_2 = None
        permute_1 = torch.ops.aten.permute.default(permute, [0, 2, 1]);  permute = None
        expand = torch.ops.aten.expand.default(getitem, [1, 8, 8]);  getitem = None
        view = torch.ops.aten.view.default(expand, [1, 8, 8]);  expand = None
        expand_1 = torch.ops.aten.expand.default(permute_1, [1, 8, 8]);  permute_1 = None
        view_1 = torch.ops.aten.view.default(expand_1, [1, 8, 8]);  expand_1 = None
        bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2 = torch.ops.aten.view.default(bmm, [1, 8, 8]);  bmm = None
        view_3 = torch.ops.aten.view.default(view_2, [1, 8, 8]);  view_2 = None
        return ()

args = [((1, 8, 8), (64, 8, 1), torch.float32, 'cuda')]
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