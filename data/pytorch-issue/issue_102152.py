import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("mosaicml/mpt-1b-redpajama-200b",
                                     trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "mosaicml/mpt-1b-redpajama-200b",
    from_tf=False,
    config=config,
    trust_remote_code=True
)

bs = 2
seqlen = 128
emb_dim = config.d_model
x = torch.randint(0,config.vocab_size,(bs,seqlen))

model = model.cuda()
inp = {"input_ids":x.cuda()}

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def evaluate(mod,inp):
    return mod(**inp)


# backend="inductor" -> FAILURE
# backend="eager" -> SUCCESS
# backend="aot_eager" -> SUCCESS
torch._dynamo.reset()
# torch._dynamo.config.verbose=True
evaluate_opt = torch.compile(evaluate, mode="max-autotune", backend="aot_eager")

print("eager:", timed(lambda: evaluate(model, inp))[1])
print("compile:", timed(lambda: evaluate_opt(model, inp))[1])

# repro.py
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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x88X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x0e\x00\x00\x00specialize_intq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x18\x00\x00\x00assume_static_by_defaultq\x0b\x89X\x10\x00\x00\x00guard_nn_modulesq\x0c\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\rc__builtin__\nset\nq\x0e]q\x0f\x85q\x10Rq\x11X\x0f\x00\x00\x00suppress_errorsq\x12\x89X\x15\x00\x00\x00replay_record_enabledq\x13\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x14\x88X\x12\x00\x00\x00print_graph_breaksq\x15\x89X\x07\x00\x00\x00disableq\x16\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x17h\x0e]q\x18(X\x0c\x00\x00\x00torch._primsq\x19X\r\x00\x00\x00torch.testingq\x1aX\r\x00\x00\x00torch._decompq\x1bX\x13\x00\x00\x00torch.distributionsq\x1cX\x0b\x00\x00\x00torch._refsq\x1de\x85q\x1eRq\x1fX\x12\x00\x00\x00repro_forward_onlyq \x89X\x0f\x00\x00\x00repro_toleranceq!G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq"\x89X \x00\x00\x00capture_dynamic_output_shape_opsq#\x89X\x19\x00\x00\x00enforce_cond_guards_matchq$\x88X\x0c\x00\x00\x00optimize_ddpq%\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq&\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq\'\x89X\x17\x00\x00\x00raise_on_backend_changeq(\x89X\x18\x00\x00\x00error_on_nested_fx_traceq)\x88X\t\x00\x00\x00allow_rnnq*\x89X\x08\x00\x00\x00base_dirq+X&\x00\x00\x00/usr/local/lib/python3.8/dist-packagesq,X\x14\x00\x00\x00profile_cache_lookupq-\x89X\x12\x00\x00\x00DEBUG_DIR_VAR_NAMEq.X\x17\x00\x00\x00TORCH_COMPILE_DEBUG_DIRq/X\x0e\x00\x00\x00debug_dir_rootq0X&\x00\x00\x00/workspace/workdir/torch_compile_debugq1X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq2\x89X\x13\x00\x00\x00_save_config_ignoreq3h\x0e]q4(X!\x00\x00\x00skipfiles_inline_module_allowlistq5X\x0b\x00\x00\x00repro_afterq6X\x12\x00\x00\x00constant_functionsq7X\x0b\x00\x00\x00repro_levelq8e\x85q9Rq:u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x15\x00\x00\x00search_autotune_cacheq\x10\x89X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x89X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\x0b\x00\x00\x00tune_layoutq\x16\x89X\x11\x00\x00\x00aggressive_fusionq\x17\x89X\x0f\x00\x00\x00max_fusion_sizeq\x18K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x19K\x08X\x0e\x00\x00\x00comment_originq\x1a\x89X\x10\x00\x00\x00benchmark_kernelq\x1b\x89X\x12\x00\x00\x00developer_warningsq\x1c\x88X\x0f\x00\x00\x00compile_threadsq\x1dK\x0cX\x11\x00\x00\x00global_cache_pathq\x1eNX\x13\x00\x00\x00kernel_name_max_opsq\x1fK\nX\r\x00\x00\x00shape_paddingq \x89X\x0e\x00\x00\x00permute_fusionq!\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq"\x89X\x18\x00\x00\x00_raise_error_for_testingq#\x89X\x0c\x00\x00\x00_profile_varq$X\x00\x00\x00\x00q%X\x11\x00\x00\x00profile_bandwidthq&\x89X\x17\x00\x00\x00profile_bandwidth_regexq\'h%X\x0b\x00\x00\x00cpp.threadsq(J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq)\x89X\x0b\x00\x00\x00cpp.simdlenq*NX\x12\x00\x00\x00cpp.min_chunk_sizeq+M\x00\x10X\x07\x00\x00\x00cpp.cxxq,NX\x03\x00\x00\x00g++q-\x86q.X\x19\x00\x00\x00cpp.enable_kernel_profileq/\x89X\x12\x00\x00\x00cpp.weight_prepackq0\x88X\x11\x00\x00\x00triton.cudagraphsq1\x89X\x17\x00\x00\x00triton.debug_sync_graphq2\x89X\x18\x00\x00\x00triton.debug_sync_kernelq3\x89X\x15\x00\x00\x00triton.dense_indexingq4\x89X\x10\x00\x00\x00triton.max_tilesq5K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq6\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq7\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq8\x88X\x1a\x00\x00\x00triton.unique_kernel_namesq9\x89X\x18\x00\x00\x00triton.descriptive_namesq:X\x04\x00\x00\x00atenq;X\x1c\x00\x00\x00triton.persistent_reductionsq<\x88X\x10\x00\x00\x00triton.max_blockq=}q>(X\x01\x00\x00\x00Xq?M\x00\x08X\x01\x00\x00\x00Yq@M\x00\x04X\x01\x00\x00\x00ZqAM\x00\x04uX\r\x00\x00\x00trace.enabledqB\x89X\x0f\x00\x00\x00trace.debug_logqC\x88X\x0e\x00\x00\x00trace.info_logqD\x89X\x0e\x00\x00\x00trace.fx_graphqE\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqF\x88X\x13\x00\x00\x00trace.ir_pre_fusionqG\x88X\x14\x00\x00\x00trace.ir_post_fusionqH\x88X\x11\x00\x00\x00trace.output_codeqI\x88X\x13\x00\x00\x00trace.graph_diagramqJ\x89X\x15\x00\x00\x00trace.compile_profileqK\x89X\x10\x00\x00\x00trace.upload_tarqLNu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x14\x00\x00\x00static_weight_shapesq\t\x88X\x03\x00\x00\x00cseq\n\x88X\x10\x00\x00\x00max_dist_from_bwq\x0bK\x03X\t\x00\x00\x00log_levelq\x0cK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.1.0a0+fe05266
# torch cuda version: 12.1
# torch git version: Unknown


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Tue_Feb__7_19:32:13_PST_2023 
# Cuda compilation tools, release 12.1, V12.1.66 
# Build cuda_12.1.r12.1/compiler.32415258_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, slice_2):
        slice_3 = torch.ops.aten.slice.Tensor(slice_2, 2, -128, 9223372036854775807);  slice_2 = None
        return (slice_3,)
        
args = [((1, 16, 1, 2048), (32768, 2048, 2048, 1), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
ref = compiled(args)
torch.cuda.synchronize() # Ensures that segfaults are surfaced

# eager: 0.027992063522338868
for i in range(10):
#     print("eager:", timed(lambda: evaluate(model, inp))[1])
    print("compile:", timed(lambda: evaluate_opt(model, inp))[1])
compile: 0.013792256355285644
compile: 0.641217529296875
compile: 0.012715007781982422
compile: 0.6281533203125
compile: 0.012806143760681152
compile: 0.6429112548828125
compile: 0.012747776031494141
compile: 0.62847900390625
compile: 0.012733440399169921
compile: 0.64538623046875

def forward(mod,inp):
    return mod(**inp)

# NOTE: better to compile full training step
forward_opt = torch.compile(forward)
forward_backward_opt = torch.compile(forward_backward)

def forward_backward(mod,inp):
    loss = mod(**inp).logits.sum()
    loss.backward()
    return loss

for i in range(10):
    print("compile[forward]:", timed(lambda: forward_opt(model_opt, inp))[1])
    print("compile[forward_backward]:", timed(lambda: forward_backward_opt(model_opt, inp))[1])

compile[forward]: 0.01377280044555664
compile[forward_backward]: 0.050108417510986325
compile[forward]: 0.012682239532470703
compile[forward_backward]: 0.04980633544921875
compile[forward]: 0.012646400451660156
compile[forward_backward]: 0.04974387359619141
compile[forward]: 0.012747776031494141
compile[forward_backward]: 0.04998553466796875
compile[forward]: 0.012411904335021973
compile[forward_backward]: 0.04342784118652344
compile[forward]: 0.010864640235900879
compile[forward_backward]: 0.04345651245117187
compile[forward]: 0.010910719871520995
compile[forward_backward]: 0.043396095275878906
compile[forward]: 0.010874879837036134
compile[forward_backward]: 0.04341657638549805
compile[forward]: 0.011084799766540527
compile[forward_backward]: 0.04323123168945313
compile[forward]: 0.010866687774658204
compile[forward_backward]: 0.04346879959106445