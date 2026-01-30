import torch.nn as nn

import torch
import torch._dynamo
torch._dynamo.reset()
import logging
TORCHDYNAMO_REPRO_AFTER="dynamo"
torch.set_float32_matmul_precision('high')

@torch.compile(mode='reduce-overhead')
def compute(log_probs):
    lit_weights = torch.stack((log_probs, log_probs), dim=-1).permute(1, 2, 0)

    levels = [torch.tensor([5, 7], device='cuda'), torch.tensor([8], device='cuda')]
    lit_indices = torch.tensor([0, 1, 2, 3, 4, 6], device='cuda')
    id = 9
    node_indices = torch.tensor([[[0, 0],[0, 0]],[[0, 0],[0, 0]],[[0, 0],[0, 0]],[[0, 0],[0, 0]],
    [[0, 0],[0, 0]],[[1, 2],[3, 4]],[[0, 0],[0, 0]],[[1, 4],[9, 9]],[[0, 5],[6, 7]]], device='cuda')
    lit_mask = (torch.tensor([0, 1, 2, 1, 2, 0], device='cuda'), torch.tensor([1, 1, 0, 0, 1, 0], device='cuda'))
    lit_indices = torch.tensor([0, 1, 2, 3, 4, 6], device='cuda')

    data = torch.zeros(id+1, 5, device='cuda')
    data[id] =  -float(1000)
    data[lit_indices] = lit_weights[lit_mask[0], lit_mask[1]]
    
    data[levels[0]] = data[node_indices[levels[0]]].sum(-2).logsumexp(-2)
    data[levels[1]] = data[node_indices[levels[1]]].sum(-2).logsumexp(-2)

    return data[levels[-1]]

for i in range(10):
    batch_size = 5
    log_probs = torch.zeros((batch_size, 10), device='cuda', requires_grad=True)
    print(compute(log_probs).exp())

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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x0e\x00\x00\x00specialize_intq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x18\x00\x00\x00assume_static_by_defaultq\x0b\x89X\x10\x00\x00\x00guard_nn_modulesq\x0c\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\rc__builtin__\nset\nq\x0e]q\x0f\x85q\x10Rq\x11X\x0f\x00\x00\x00suppress_errorsq\x12\x89X\x15\x00\x00\x00replay_record_enabledq\x13\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x14\x88X\x12\x00\x00\x00print_graph_breaksq\x15\x89X\x07\x00\x00\x00disableq\x16\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x17h\x0e]q\x18(X\r\x00\x00\x00torch._decompq\x19X\r\x00\x00\x00torch.testingq\x1aX\x0c\x00\x00\x00torch._primsq\x1bX\x0b\x00\x00\x00torch._refsq\x1cX\x13\x00\x00\x00torch.distributionsq\x1de\x85q\x1eRq\x1fX\x12\x00\x00\x00repro_forward_onlyq \x89X\x0f\x00\x00\x00repro_toleranceq!G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq"\x89X \x00\x00\x00capture_dynamic_output_shape_opsq#\x89X\x19\x00\x00\x00enforce_cond_guards_matchq$\x88X\x0c\x00\x00\x00optimize_ddpq%\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq&\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq\'\x89X\x17\x00\x00\x00raise_on_backend_changeq(\x89X\x18\x00\x00\x00error_on_nested_fx_traceq)\x88X\t\x00\x00\x00allow_rnnq*\x89X\x08\x00\x00\x00base_dirq+XH\x00\x00\x00/space/ahmedk/anaconda3/envs/simple_updated/lib/python3.10/site-packagesq,X\x0e\x00\x00\x00debug_dir_rootq-X;\x00\x00\x00/scratch/ahmedk/simple-graphs/exactly-k/torch_compile_debugq.X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq/\x89X\x13\x00\x00\x00_save_config_ignoreq0h\x0e]q1(X\x0b\x00\x00\x00repro_afterq2X\x0b\x00\x00\x00repro_levelq3X!\x00\x00\x00skipfiles_inline_module_allowlistq4X\x12\x00\x00\x00constant_functionsq5e\x85q6Rq7u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x15\x00\x00\x00search_autotune_cacheq\x10\x89X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x89X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\x0b\x00\x00\x00tune_layoutq\x16\x89X\x11\x00\x00\x00aggressive_fusionq\x17\x89X\x0f\x00\x00\x00max_fusion_sizeq\x18K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x19K\x08X\x0e\x00\x00\x00comment_originq\x1a\x89X\x10\x00\x00\x00benchmark_kernelq\x1b\x89X\x12\x00\x00\x00developer_warningsq\x1c\x88X\x0f\x00\x00\x00compile_threadsq\x1dK X\x11\x00\x00\x00global_cache_pathq\x1eNX\x13\x00\x00\x00kernel_name_max_opsq\x1fK\nX\r\x00\x00\x00shape_paddingq \x89X\x0e\x00\x00\x00permute_fusionq!\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq"\x89X\x18\x00\x00\x00_raise_error_for_testingq#\x89X\x0c\x00\x00\x00_profile_varq$X\x00\x00\x00\x00q%X\x11\x00\x00\x00profile_bandwidthq&\x89X\x17\x00\x00\x00profile_bandwidth_regexq\'h%X\x0b\x00\x00\x00cpp.threadsq(J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq)\x89X\x0b\x00\x00\x00cpp.simdlenq*NX\x12\x00\x00\x00cpp.min_chunk_sizeq+M\x00\x10X\x07\x00\x00\x00cpp.cxxq,NX\x03\x00\x00\x00g++q-\x86q.X\x19\x00\x00\x00cpp.enable_kernel_profileq/\x89X\x12\x00\x00\x00cpp.weight_prepackq0\x88X\x11\x00\x00\x00triton.cudagraphsq1\x89X\x17\x00\x00\x00triton.debug_sync_graphq2\x89X\x18\x00\x00\x00triton.debug_sync_kernelq3\x89X\x15\x00\x00\x00triton.dense_indexingq4\x89X\x10\x00\x00\x00triton.max_tilesq5K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq6\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq7\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq8\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq9\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq:\x89X\x1c\x00\x00\x00triton.persistent_reductionsq;\x88X\x10\x00\x00\x00triton.max_blockq<}q=(X\x01\x00\x00\x00Xq>M\x00\x08X\x01\x00\x00\x00Yq?M\x00\x04X\x01\x00\x00\x00Zq@M\x00\x04uX\r\x00\x00\x00trace.enabledqA\x89X\x0f\x00\x00\x00trace.debug_logqB\x88X\x0e\x00\x00\x00trace.info_logqC\x89X\x0e\x00\x00\x00trace.fx_graphqD\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqE\x88X\x13\x00\x00\x00trace.ir_pre_fusionqF\x88X\x14\x00\x00\x00trace.ir_post_fusionqG\x88X\x11\x00\x00\x00trace.output_codeqH\x88X\x13\x00\x00\x00trace.graph_diagramqI\x89X\x15\x00\x00\x00trace.compile_profileqJ\x89X\x10\x00\x00\x00trace.upload_tarqKNu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.1.0.dev20230304+cu117
# torch cuda version: 11.7
# torch git version: f1d60f587239a2abdb913619faa61e81ee393ecc


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2021 NVIDIA Corporation 
# Built on Thu_Nov_18_09:45:30_PST_2021 
# Cuda compilation tools, release 11.5, V11.5.119 
# Build cuda_11.5.r11.5/compiler.30672275_0 

# GPU Hardware Info: 
# NVIDIA RTX A5000 : 4 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_tensor_constant0', torch.randint(1, size=[2], dtype=torch.int64).cuda())
        self.register_buffer('_tensor_constant1', torch.randint(1, size=[1], dtype=torch.int64).cuda())
        self.register_buffer('_tensor_constant3', torch.randint(1, size=[9, 2, 2], dtype=torch.int64).cuda())
        self.register_buffer('_tensor_constant4', torch.randint(1, size=[6], dtype=torch.int64).cuda())
        self.register_buffer('_tensor_constant5', torch.randint(1, size=[6], dtype=torch.int64).cuda())
        self.register_buffer('_tensor_constant6', torch.randint(1, size=[6], dtype=torch.int64).cuda())



    def forward(self, primals_1):
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_1, 2);  primals_1 = None
        cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze], -1);  unsqueeze = None
        permute = torch.ops.aten.permute.default(cat, [1, 2, 0]);  cat = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        _tensor_constant3 = self._tensor_constant3
        lift_fresh_copy_3 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        _tensor_constant4 = self._tensor_constant4
        lift_fresh_copy_4 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
        _tensor_constant5 = self._tensor_constant5
        lift_fresh_copy_5 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
        _tensor_constant6 = self._tensor_constant6
        lift_fresh_copy_6 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
        full = torch.ops.aten.full.default([10, 5], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda'), pin_memory = False)
        select = torch.ops.aten.select.int(full, 0, 9)
        full_like = torch.ops.aten.full_like.default(select, -1000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  select = None
        select_scatter = torch.ops.aten.select_scatter.default(full, full_like, 0, 9);  full = full_like = None
        index = torch.ops.aten.index.Tensor(permute, [lift_fresh_copy_4, lift_fresh_copy_5]);  permute = lift_fresh_copy_4 = lift_fresh_copy_5 = None
        index_put = torch.ops.aten.index_put.default(select_scatter, [lift_fresh_copy_6], index);  select_scatter = lift_fresh_copy_6 = index = None
        index_1 = torch.ops.aten.index.Tensor(lift_fresh_copy_3, [lift_fresh_copy])
        index_2 = torch.ops.aten.index.Tensor(index_put, [index_1])
        sum_1 = torch.ops.aten.sum.dim_IntList(index_2, [-2]);  index_2 = None
        amax = torch.ops.aten.amax.default(sum_1, [-2], True)
        abs_1 = torch.ops.aten.abs.default(amax)
        eq = torch.ops.aten.eq.Scalar(abs_1, inf);  abs_1 = None
        scalar_tensor = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where = torch.ops.aten.where.self(eq, scalar_tensor, amax);  eq = amax = None
        squeeze = torch.ops.aten.squeeze.dims(where, [-2])
        sub = torch.ops.aten.sub.Tensor(sum_1, where);  where = None
        exp = torch.ops.aten.exp.default(sub);  sub = None
        sum_2 = torch.ops.aten.sum.dim_IntList(exp, [-2]);  exp = None
        log = torch.ops.aten.log.default(sum_2);  sum_2 = None
        add = torch.ops.aten.add.Tensor(log, squeeze);  log = squeeze = None
        index_put_1 = torch.ops.aten.index_put.default(index_put, [lift_fresh_copy], add);  index_put = lift_fresh_copy = None
        index_3 = torch.ops.aten.index.Tensor(lift_fresh_copy_3, [lift_fresh_copy_1]);  lift_fresh_copy_3 = None
        index_4 = torch.ops.aten.index.Tensor(index_put_1, [index_3])
        sum_3 = torch.ops.aten.sum.dim_IntList(index_4, [-2]);  index_4 = None
        amax_1 = torch.ops.aten.amax.default(sum_3, [-2], True)
        abs_2 = torch.ops.aten.abs.default(amax_1)
        eq_1 = torch.ops.aten.eq.Scalar(abs_2, inf);  abs_2 = None
        where_1 = torch.ops.aten.where.self(eq_1, scalar_tensor, amax_1);  eq_1 = scalar_tensor = amax_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(where_1, [-2])
        sub_1 = torch.ops.aten.sub.Tensor(sum_3, where_1);  where_1 = None
        exp_1 = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_1, [-2]);  exp_1 = None
        log_1 = torch.ops.aten.log.default(sum_4);  sum_4 = None
        add_1 = torch.ops.aten.add.Tensor(log_1, squeeze_1);  log_1 = squeeze_1 = None
        view = torch.ops.aten.view.default(add_1, [5])
        index_put_2 = torch.ops.aten.index_put.default(index_put_1, [lift_fresh_copy_1], view);  index_put_1 = view = None
        index_5 = torch.ops.aten.index.Tensor(index_put_2, [lift_fresh_copy_1]);  index_put_2 = lift_fresh_copy_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(add_1, -2);  add_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(sum_3, unsqueeze_3);  sum_3 = unsqueeze_3 = None
        exp_2 = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(add, -2);  add = None
        sub_3 = torch.ops.aten.sub.Tensor(sum_1, unsqueeze_6);  sum_1 = unsqueeze_6 = None
        exp_3 = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        return [index_5, index_1, index_3, exp_2, exp_3]

args = [((5, 10), (10, 1), torch.float32, 'cuda')]
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
    module_fails=partial(isolate_fails, env=env_variables, compiler_name="inductor_accuracy", patch_code=isolate_fails_code_str),
    dump_state=partial(dump_compiler_graph_state, compiler_name="inductor_accuracy"),
)

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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x18\x00\x00\x00assume_static_by_defaultq\x0b\x89X\x10\x00\x00\x00guard_nn_modulesq\x0c\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\rc__builtin__\nset\nq\x0e]q\x0f\x85q\x10Rq\x11X\x0f\x00\x00\x00suppress_errorsq\x12\x89X\x15\x00\x00\x00replay_record_enabledq\x13\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x14\x88X\x12\x00\x00\x00print_graph_breaksq\x15\x89X\x07\x00\x00\x00disableq\x16\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x17h\x0e]q\x18(X\x0b\x00\x00\x00torch._refsq\x19X\x0c\x00\x00\x00torch._primsq\x1aX\x13\x00\x00\x00torch.distributionsq\x1bX\r\x00\x00\x00torch.testingq\x1cX\r\x00\x00\x00torch._decompq\x1de\x85q\x1eRq\x1fX\x12\x00\x00\x00repro_forward_onlyq \x89X\x0f\x00\x00\x00repro_toleranceq!G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq"\x89X \x00\x00\x00capture_dynamic_output_shape_opsq#\x89X\x19\x00\x00\x00enforce_cond_guards_matchq$\x88X\x0c\x00\x00\x00optimize_ddpq%\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq&\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq\'\x89X\x17\x00\x00\x00raise_on_backend_changeq(\x89X\x18\x00\x00\x00error_on_nested_fx_traceq)\x88X\t\x00\x00\x00allow_rnnq*\x89X\x08\x00\x00\x00base_dirq+XH\x00\x00\x00/space/ahmedk/anaconda3/envs/simple_updated/lib/python3.10/site-packagesq,X\x0e\x00\x00\x00debug_dir_rootq-X;\x00\x00\x00/scratch/ahmedk/simple-graphs/exactly-k/torch_compile_debugq.X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq/\x89X\x13\x00\x00\x00_save_config_ignoreq0h\x0e]q1(X\x0b\x00\x00\x00repro_levelq2X!\x00\x00\x00skipfiles_inline_module_allowlistq3X\x12\x00\x00\x00constant_functionsq4X\x0b\x00\x00\x00repro_afterq5e\x85q6Rq7X\x0e\x00\x00\x00specialize_intq8\x88u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x15\x00\x00\x00search_autotune_cacheq\x10\x89X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x89X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\x0b\x00\x00\x00tune_layoutq\x16\x89X\x11\x00\x00\x00aggressive_fusionq\x17\x89X\x0f\x00\x00\x00max_fusion_sizeq\x18K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x19K\x08X\x0e\x00\x00\x00comment_originq\x1a\x89X\x12\x00\x00\x00developer_warningsq\x1b\x88X\x0f\x00\x00\x00compile_threadsq\x1cK X\x11\x00\x00\x00global_cache_pathq\x1dNX\x13\x00\x00\x00kernel_name_max_opsq\x1eK\nX\r\x00\x00\x00shape_paddingq\x1f\x89X\x0e\x00\x00\x00permute_fusionq \x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq!\x89X\x18\x00\x00\x00_raise_error_for_testingq"\x89X\x0c\x00\x00\x00_profile_varq#X\x00\x00\x00\x00q$X\x11\x00\x00\x00profile_bandwidthq%\x89X\x17\x00\x00\x00profile_bandwidth_regexq&h$X\x0b\x00\x00\x00cpp.threadsq\'J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq(\x89X\x0b\x00\x00\x00cpp.simdlenq)NX\x12\x00\x00\x00cpp.min_chunk_sizeq*M\x00\x10X\x07\x00\x00\x00cpp.cxxq+NX\x03\x00\x00\x00g++q,\x86q-X\x19\x00\x00\x00cpp.enable_kernel_profileq.\x89X\x12\x00\x00\x00cpp.weight_prepackq/\x88X\x11\x00\x00\x00triton.cudagraphsq0\x89X\x17\x00\x00\x00triton.debug_sync_graphq1\x89X\x18\x00\x00\x00triton.debug_sync_kernelq2\x89X\x12\x00\x00\x00triton.convolutionq3X\x04\x00\x00\x00atenq4X\x15\x00\x00\x00triton.dense_indexingq5\x89X\x10\x00\x00\x00triton.max_tilesq6K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq7\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq8\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq9\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq:\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq;\x89X\x1c\x00\x00\x00triton.persistent_reductionsq<\x88X\x10\x00\x00\x00triton.max_blockq=}q>(X\x01\x00\x00\x00Xq?M\x00\x08X\x01\x00\x00\x00Yq@M\x00\x04X\x01\x00\x00\x00ZqAM\x00\x04uX\r\x00\x00\x00trace.enabledqB\x89X\x0f\x00\x00\x00trace.debug_logqC\x88X\x0e\x00\x00\x00trace.info_logqD\x89X\x0e\x00\x00\x00trace.fx_graphqE\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqF\x88X\x13\x00\x00\x00trace.ir_pre_fusionqG\x88X\x14\x00\x00\x00trace.ir_post_fusionqH\x88X\x11\x00\x00\x00trace.output_codeqI\x88X\x13\x00\x00\x00trace.graph_diagramqJ\x89X\x15\x00\x00\x00trace.compile_profileqK\x89X\x10\x00\x00\x00trace.upload_tarqLNX\x10\x00\x00\x00benchmark_kernelqM\x89u.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.0a0+gitfafb410
# torch cuda version: 11.8
# torch git version: fafb410985d2cb94bd95f12f0c392bad9385b643


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Wed_Sep_21_10:33:58_PDT_2022 
# Cuda compilation tools, release 11.8, V11.8.89 
# Build cuda_11.8.r11.8/compiler.31833905_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-40GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, lift_fresh_copy_1, index_put_1, view):
        index_put_2 = torch.ops.aten.index_put.default(index_put_1, [lift_fresh_copy_1], view);  index_put_1 = view = None
        index_5 = torch.ops.aten.index.Tensor(index_put_2, [lift_fresh_copy_1]);  index_put_2 = lift_fresh_copy_1 = None
        return [index_5]
        
args = [((1,), (1,), torch.int64, 'cuda'), ((10, 5), (5, 1), torch.float32, 'cuda'), ((5,), (1,), torch.float32, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro(), tracing_mode='real')(*args)

from torch._inductor.compile_fx import compile_fx_inner
from torch._dynamo.debug_utils import same_two_models

compiled = compile_fx_inner(mod, args)
class AccuracyError(Exception):
    pass
if not same_two_models(mod, compiled, args, only_fwd=True):
    raise AccuracyError("Bad accuracy detected")