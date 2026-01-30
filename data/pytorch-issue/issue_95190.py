import torch.nn as nn

def loss(self, example, preds_dicts, test_cfg, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            # preds_dict['hm'] = self._sigmoid(preds_dict['hm'])
            preds_dict['hm'] = torch.clamp(torch.sigmoid(preds_dict['hm']), min=1e-4, max=1-1e-4)
                
            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])

            target_box = example['anno_box'][task_id]
            # reconstruct the anno_box from multiple reg heads
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dict:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['vel'], preds_dict['rot']), dim=1)  
                else:
                    preds_dict['anno_box'] = torch.cat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                        preds_dict['rot']), dim=1)   
                    target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]] # remove vel target                       
            else:
                raise NotImplementedError()

            ret = {}
 
            # Regression loss for dimension, offset, height, rotation            
            box_loss = self.crit_reg(preds_dict['anno_box'], example['mask'][task_id], example['ind'][task_id], target_box)

            loc_loss = (box_loss*box_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight*loc_loss

            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': box_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})

            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

import os
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import functools
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.testing import rand_strided

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\x0c\x00\x00\x00torch._primsq\x18X\r\x00\x00\x00torch._decompq\x19X\x13\x00\x00\x00torch.distributionsq\x1aX\x0b\x00\x00\x00torch._refsq\x1bX\r\x00\x00\x00torch.testingq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x18\x00\x00\x00error_on_nested_fx_traceq&\x88X\t\x00\x00\x00allow_rnnq\'\x89X\x08\x00\x00\x00base_dirq(XA\x00\x00\x00/opt/conda/envs/centerpoint_torch1.14/lib/python3.8/site-packagesq)X\x0e\x00\x00\x00debug_dir_rootq*XA\x00\x00\x00/home/users/chenrui17/3d_models/cp-torch_1_14/torch_compile_debugq+X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq,\x89X\x13\x00\x00\x00_save_config_ignoreq-h\r]q.(X!\x00\x00\x00skipfiles_inline_module_allowlistq/X\x12\x00\x00\x00constant_functionsq0X\x0b\x00\x00\x00repro_afterq1X\x0b\x00\x00\x00repro_levelq2e\x85q3Rq4u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x12\x00\x00\x00developer_warningsq\x02\x88X\x10\x00\x00\x00disable_progressq\x03\x88X\x10\x00\x00\x00verbose_progressq\x04\x89X\x0b\x00\x00\x00cpp_wrapperq\x05\x89X\x03\x00\x00\x00dceq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x0c\x00\x00\x00size_assertsq\x08\x88X\x10\x00\x00\x00pick_loop_ordersq\t\x88X\x0f\x00\x00\x00inplace_buffersq\n\x88X\x11\x00\x00\x00benchmark_harnessq\x0b\x88X\x0f\x00\x00\x00epilogue_fusionq\x0c\x89X\x15\x00\x00\x00epilogue_fusion_firstq\r\x89X\x0f\x00\x00\x00pattern_matcherq\x0e\x88X\n\x00\x00\x00reorderingq\x0f\x89X\x0c\x00\x00\x00max_autotuneq\x10\x89X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x89X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\r\x00\x00\x00prefuse_nodesq\x16\x88X\x0b\x00\x00\x00tune_layoutq\x17\x89X\x11\x00\x00\x00aggressive_fusionq\x18\x89X\x0f\x00\x00\x00max_fusion_sizeq\x19K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x1aK\x08X\x0e\x00\x00\x00comment_originq\x1b\x89X\x0f\x00\x00\x00compile_threadsq\x1cK X\x13\x00\x00\x00kernel_name_max_opsq\x1dK\nX\r\x00\x00\x00shape_paddingq\x1e\x89X\x0e\x00\x00\x00permute_fusionq\x1f\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq \x89X\x0b\x00\x00\x00cpp.threadsq!J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq"\x89X\x0b\x00\x00\x00cpp.simdlenq#NX\x12\x00\x00\x00cpp.min_chunk_sizeq$M\x00\x10X\x07\x00\x00\x00cpp.cxxq%NX\x03\x00\x00\x00g++q&\x86q\'X\x19\x00\x00\x00cpp.enable_kernel_profileq(\x89X\x12\x00\x00\x00cpp.weight_prepackq)\x88X\x11\x00\x00\x00triton.cudagraphsq*\x89X\x17\x00\x00\x00triton.debug_sync_graphq+\x89X\x18\x00\x00\x00triton.debug_sync_kernelq,\x89X\x12\x00\x00\x00triton.convolutionq-X\x04\x00\x00\x00atenq.X\x15\x00\x00\x00triton.dense_indexingq/\x89X\x10\x00\x00\x00triton.max_tilesq0K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq1\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq2\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq3\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq4\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq5\x89X\x1c\x00\x00\x00triton.persistent_reductionsq6\x89X\r\x00\x00\x00trace.enabledq7\x89X\x0f\x00\x00\x00trace.debug_logq8\x88X\x0e\x00\x00\x00trace.info_logq9\x89X\x0e\x00\x00\x00trace.fx_graphq:\x88X\x1a\x00\x00\x00trace.fx_graph_transformedq;\x88X\x13\x00\x00\x00trace.ir_pre_fusionq<\x88X\x14\x00\x00\x00trace.ir_post_fusionq=\x88X\x11\x00\x00\x00trace.output_codeq>\x88X\x13\x00\x00\x00trace.graph_diagramq?\x89X\x15\x00\x00\x00trace.compile_profileq@\x89X\x10\x00\x00\x00trace.upload_tarqANu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


args = [((16, 1, 128, 128), (16384, 16384, 128, 1), torch.float16, 'cuda', True), ((16, 1, 128, 128), (16384, 16384, 128, 1), torch.float32, 'cuda', False), ((16, 500), (500, 1), torch.int64, 'cuda', False), ((16, 500), (500, 1), torch.uint8, 'cuda', False), ((16, 500), (500, 1), torch.int64, 'cuda', False)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, out : torch.Tensor, target : torch.Tensor, ind : torch.Tensor, mask : torch.Tensor, cat : torch.Tensor):
        float_1 = mask.float();  mask = None
        sub = 1 - target;  target = None
        pow_1 = torch.pow(sub, 4);  sub = None
        sub_1 = 1 - out
        log = torch.log(sub_1);  sub_1 = None
        pow_2 = torch.pow(out, 2)
        mul = log * pow_2;  log = pow_2 = None
        mul_1 = mul * pow_1;  mul = pow_1 = None
        sum_1 = mul_1.sum();  mul_1 = None
        permute = out.permute(0, 2, 3, 1);  out = None
        contiguous = permute.contiguous();  permute = None
        view = contiguous.view(16, -1, 1);  contiguous = None
        unsqueeze = ind.unsqueeze(2);  ind = None
        expand = unsqueeze.expand(16, 500, 1);  unsqueeze = None
        gather = view.gather(1, expand);  view = expand = None
        unsqueeze_1 = cat.unsqueeze(2);  cat = None
        gather_1 = gather.gather(2, unsqueeze_1);  gather = unsqueeze_1 = None
        sum_2 = float_1.sum()
        log_1 = torch.log(gather_1)
        sub_2 = 1 - gather_1;  gather_1 = None
        pow_3 = torch.pow(sub_2, 2);  sub_2 = None
        mul_2 = log_1 * pow_3;  log_1 = pow_3 = None
        unsqueeze_2 = float_1.unsqueeze(2);  float_1 = None
        mul_3 = mul_2 * unsqueeze_2;  mul_2 = unsqueeze_2 = None
        sum_3 = mul_3.sum();  mul_3 = None
        eq = sum_2 == 0
        return (sum_3, sum_2, sum_1, eq)


mod = Repro()

# Setup debug minifier compiler
torch._dynamo.debug_utils.MINIFIER_SPAWNED = True
compiler_fn = lookup_backend("dynamo_minifier_backend")

dynamo_minifier_backend = functools.partial(
    compiler_fn,
    compiler_name="inductor",
)
opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

with torch.cuda.amp.autocast(enabled=True):
    opt_mod(*args)

import os
from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import functools
import torch._dynamo
from torch._dynamo.debug_utils import run_fwd_maybe_bwd
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.testing import rand_strided

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\r\x00\x00\x00torch.testingq\x18X\x0c\x00\x00\x00torch._primsq\x19X\x0b\x00\x00\x00torch._refsq\x1aX\x13\x00\x00\x00torch.distributionsq\x1bX\r\x00\x00\x00torch._decompq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x18\x00\x00\x00error_on_nested_fx_traceq&\x88X\t\x00\x00\x00allow_rnnq\'\x89X\x08\x00\x00\x00base_dirq(XA\x00\x00\x00/opt/conda/envs/centerpoint_torch1.14/lib/python3.8/site-packagesq)X\x0e\x00\x00\x00debug_dir_rootq*XI\x00\x00\x00/home/users/chenrui17/3d_models/cp-torch-2.0-original/torch_compile_debugq+X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq,\x89X\x13\x00\x00\x00_save_config_ignoreq-h\r]q.(X!\x00\x00\x00skipfiles_inline_module_allowlistq/X\x0b\x00\x00\x00repro_levelq0X\x12\x00\x00\x00constant_functionsq1X\x0b\x00\x00\x00repro_afterq2e\x85q3Rq4u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x12\x00\x00\x00developer_warningsq\x02\x88X\x10\x00\x00\x00disable_progressq\x03\x88X\x10\x00\x00\x00verbose_progressq\x04\x89X\x0b\x00\x00\x00cpp_wrapperq\x05\x89X\x03\x00\x00\x00dceq\x06\x89X\x14\x00\x00\x00static_weight_shapesq\x07\x88X\x0c\x00\x00\x00size_assertsq\x08\x88X\x10\x00\x00\x00pick_loop_ordersq\t\x88X\x0f\x00\x00\x00inplace_buffersq\n\x88X\x11\x00\x00\x00benchmark_harnessq\x0b\x88X\x0f\x00\x00\x00epilogue_fusionq\x0c\x89X\x15\x00\x00\x00epilogue_fusion_firstq\r\x89X\x0f\x00\x00\x00pattern_matcherq\x0e\x88X\n\x00\x00\x00reorderingq\x0f\x89X\x0c\x00\x00\x00max_autotuneq\x10\x89X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x89X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\r\x00\x00\x00prefuse_nodesq\x16\x88X\x0b\x00\x00\x00tune_layoutq\x17\x89X\x11\x00\x00\x00aggressive_fusionq\x18\x89X\x0f\x00\x00\x00max_fusion_sizeq\x19K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x1aK\x08X\x0e\x00\x00\x00comment_originq\x1b\x89X\x0f\x00\x00\x00compile_threadsq\x1cK X\x13\x00\x00\x00kernel_name_max_opsq\x1dK\nX\r\x00\x00\x00shape_paddingq\x1e\x89X\x0e\x00\x00\x00permute_fusionq\x1f\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq \x89X\x0b\x00\x00\x00cpp.threadsq!J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq"\x89X\x0b\x00\x00\x00cpp.simdlenq#NX\x12\x00\x00\x00cpp.min_chunk_sizeq$M\x00\x10X\x07\x00\x00\x00cpp.cxxq%NX\x03\x00\x00\x00g++q&\x86q\'X\x19\x00\x00\x00cpp.enable_kernel_profileq(\x89X\x12\x00\x00\x00cpp.weight_prepackq)\x88X\x11\x00\x00\x00triton.cudagraphsq*\x89X\x17\x00\x00\x00triton.debug_sync_graphq+\x89X\x18\x00\x00\x00triton.debug_sync_kernelq,\x89X\x12\x00\x00\x00triton.convolutionq-X\x04\x00\x00\x00atenq.X\x15\x00\x00\x00triton.dense_indexingq/\x89X\x10\x00\x00\x00triton.max_tilesq0K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq1\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq2\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq3\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq4\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq5\x89X\x1c\x00\x00\x00triton.persistent_reductionsq6\x89X\r\x00\x00\x00trace.enabledq7\x89X\x0f\x00\x00\x00trace.debug_logq8\x88X\x0e\x00\x00\x00trace.info_logq9\x89X\x0e\x00\x00\x00trace.fx_graphq:\x88X\x1a\x00\x00\x00trace.fx_graph_transformedq;\x88X\x13\x00\x00\x00trace.ir_pre_fusionq<\x88X\x14\x00\x00\x00trace.ir_post_fusionq=\x88X\x11\x00\x00\x00trace.output_codeq>\x88X\x13\x00\x00\x00trace.graph_diagramq?\x89X\x15\x00\x00\x00trace.compile_profileq@\x89X\x10\x00\x00\x00trace.upload_tarqANu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


args = [((16, 1, 128, 128), (16384, 16384, 128, 1), torch.float32, 'cuda', True)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, x : torch.Tensor):
        sigmoid_ = x.sigmoid_();  x = None
        clamp = torch.clamp(sigmoid_, min = 0.0001, max = 0.9999);  sigmoid_ = None
        return (clamp,)


mod = Repro()

# Setup debug minifier compiler
torch._dynamo.debug_utils.MINIFIER_SPAWNED = True
compiler_fn = lookup_backend("dynamo_minifier_backend")

dynamo_minifier_backend = functools.partial(
    compiler_fn,
    compiler_name="inductor",
)
opt_mod = torch._dynamo.optimize(dynamo_minifier_backend)(mod)

with torch.cuda.amp.autocast(enabled=False):
    opt_mod(*args)