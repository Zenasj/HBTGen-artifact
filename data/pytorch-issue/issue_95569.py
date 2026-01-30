import torch.nn as nn

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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x88X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x18\x00\x00\x00assume_static_by_defaultq\x0b\x89X\x10\x00\x00\x00guard_nn_modulesq\x0c\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\rc__builtin__\nset\nq\x0e]q\x0f\x85q\x10Rq\x11X\x0f\x00\x00\x00suppress_errorsq\x12\x89X\x15\x00\x00\x00replay_record_enabledq\x13\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x14\x88X\x12\x00\x00\x00print_graph_breaksq\x15\x89X\x07\x00\x00\x00disableq\x16\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x17h\x0e]q\x18(X\r\x00\x00\x00torch.testingq\x19X\x0b\x00\x00\x00torch._refsq\x1aX\x13\x00\x00\x00torch.distributionsq\x1bX\r\x00\x00\x00torch._decompq\x1cX\x0c\x00\x00\x00torch._primsq\x1de\x85q\x1eRq\x1fX\x12\x00\x00\x00repro_forward_onlyq \x89X\x0f\x00\x00\x00repro_toleranceq!G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq"\x89X \x00\x00\x00capture_dynamic_output_shape_opsq#\x89X\x19\x00\x00\x00enforce_cond_guards_matchq$\x88X\x0c\x00\x00\x00optimize_ddpq%\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq&\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq\'\x89X\x17\x00\x00\x00raise_on_backend_changeq(\x89X\x18\x00\x00\x00error_on_nested_fx_traceq)\x88X\t\x00\x00\x00allow_rnnq*\x89X\x08\x00\x00\x00base_dirq+X\x08\x00\x00\x00minifiedq,X\x0e\x00\x00\x00debug_dir_rootq-XR\x00\x00\x00/net/vast-storage/scratch/vast/tenenbaum/gua/Documents/routing/torch_compile_debugq.X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq/\x89X\x13\x00\x00\x00_save_config_ignoreq0h\x0e]q1(X\x12\x00\x00\x00constant_functionsq2X\x0b\x00\x00\x00repro_afterq3X!\x00\x00\x00skipfiles_inline_module_allowlistq4X\x0b\x00\x00\x00repro_levelq5e\x85q6Rq7u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x15\x00\x00\x00search_autotune_cacheq\x10\x88X\x17\x00\x00\x00realize_reads_thresholdq\x11K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x12M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x13K\x08X\x0f\x00\x00\x00fallback_randomq\x14\x89X\x12\x00\x00\x00implicit_fallbacksq\x15\x88X\x0b\x00\x00\x00tune_layoutq\x16\x89X\x11\x00\x00\x00aggressive_fusionq\x17\x89X\x0f\x00\x00\x00max_fusion_sizeq\x18K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x19K\x08X\x0e\x00\x00\x00comment_originq\x1a\x89X\x12\x00\x00\x00developer_warningsq\x1b\x88X\x0f\x00\x00\x00compile_threadsq\x1cK X\x11\x00\x00\x00global_cache_pathq\x1dNX\x13\x00\x00\x00kernel_name_max_opsq\x1eK\nX\r\x00\x00\x00shape_paddingq\x1f\x89X\x0e\x00\x00\x00permute_fusionq \x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq!\x89X\x18\x00\x00\x00_raise_error_for_testingq"\x89X\x0c\x00\x00\x00_profile_varq#X\x00\x00\x00\x00q$X\x11\x00\x00\x00profile_bandwidthq%\x89X\x17\x00\x00\x00profile_bandwidth_regexq&h$X\x0b\x00\x00\x00cpp.threadsq\'J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq(\x89X\x0b\x00\x00\x00cpp.simdlenq)NX\x12\x00\x00\x00cpp.min_chunk_sizeq*M\x00\x10X\x07\x00\x00\x00cpp.cxxq+NX\x03\x00\x00\x00g++q,\x86q-X\x19\x00\x00\x00cpp.enable_kernel_profileq.\x89X\x12\x00\x00\x00cpp.weight_prepackq/\x88X\x11\x00\x00\x00triton.cudagraphsq0\x89X\x17\x00\x00\x00triton.debug_sync_graphq1\x89X\x18\x00\x00\x00triton.debug_sync_kernelq2\x89X\x12\x00\x00\x00triton.convolutionq3X\x04\x00\x00\x00atenq4X\x15\x00\x00\x00triton.dense_indexingq5\x89X\x10\x00\x00\x00triton.max_tilesq6K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq7\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq8\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq9\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq:\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq;\x89X\x1c\x00\x00\x00triton.persistent_reductionsq<\x88X\x10\x00\x00\x00triton.max_blockq=}q>(X\x01\x00\x00\x00Xq?M\x00\x08X\x01\x00\x00\x00Yq@M\x00\x04X\x01\x00\x00\x00ZqAM\x00\x04uX\r\x00\x00\x00trace.enabledqB\x89X\x0f\x00\x00\x00trace.debug_logqC\x88X\x0e\x00\x00\x00trace.info_logqD\x89X\x0e\x00\x00\x00trace.fx_graphqE\x88X\x1a\x00\x00\x00trace.fx_graph_transformedqF\x88X\x13\x00\x00\x00trace.ir_pre_fusionqG\x88X\x14\x00\x00\x00trace.ir_post_fusionqH\x88X\x11\x00\x00\x00trace.output_codeqI\x88X\x13\x00\x00\x00trace.graph_diagramqJ\x89X\x15\x00\x00\x00trace.compile_profileqK\x89X\x10\x00\x00\x00trace.upload_tarqLNu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


args = [((1024, 20, 2), (40, 2, 1), torch.float32, 'cuda', False), ((1024, 20), (20, 1), torch.float32, 'cuda', False), ((1024, 2), (2, 1), torch.float32, 'cuda', False)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_init_embed_depot = Linear(in_features=2, out_features=128, bias=True).cuda()
        self.self_init_embed = Linear(in_features=3, out_features=128, bias=True).cuda()
        self.getattr_getattr_self_embedder_layers___0_____1___normalizer = BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.getattr_getattr_self_embedder_layers___0_____2___module_0 = Linear(in_features=128, out_features=512, bias=True).cuda()
        self.getattr_getattr_self_embedder_layers___0_____2___module_1 = ReLU()
        self.getattr_getattr_self_embedder_layers___0_____2___module_2 = Linear(in_features=512, out_features=128, bias=True).cuda()
        self.getattr_getattr_self_embedder_layers___0_____3___normalizer = BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.getattr_getattr_self_embedder_layers___1_____1___normalizer = BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.getattr_getattr_self_embedder_layers___1_____2___module_0 = Linear(in_features=128, out_features=512, bias=True).cuda()
        self.getattr_getattr_self_embedder_layers___1_____2___module_1 = ReLU()
        self.getattr_getattr_self_embedder_layers___1_____2___module_2 = Linear(in_features=512, out_features=128, bias=True).cuda()
        self.getattr_getattr_self_embedder_layers___1_____3___normalizer = BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.getattr_getattr_self_embedder_layers___2_____1___normalizer = BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.getattr_getattr_self_embedder_layers___2_____2___module_0 = Linear(in_features=128, out_features=512, bias=True).cuda()
        self.getattr_getattr_self_embedder_layers___2_____2___module_1 = ReLU()
        self.getattr_getattr_self_embedder_layers___2_____2___module_2 = Linear(in_features=512, out_features=128, bias=True).cuda()
        self.getattr_getattr_self_embedder_layers___2_____3___normalizer = BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.getattr_getattr_self_embedder_layers___0_____0___module_W_query = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___0_____0___module_W_key = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___0_____0___module_W_val = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___0_____0___module_W_out = torch.nn.Parameter(torch.randn([8, 16, 128], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___1_____0___module_W_query = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___1_____0___module_W_key = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___1_____0___module_W_val = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___1_____0___module_W_out = torch.nn.Parameter(torch.randn([8, 16, 128], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___2_____0___module_W_query = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___2_____0___module_W_key = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___2_____0___module_W_val = torch.nn.Parameter(torch.randn([8, 128, 16], dtype=torch.float32)).cuda()
        self.getattr_getattr_self_embedder_layers___2_____0___module_W_out = torch.nn.Parameter(torch.randn([8, 16, 128], dtype=torch.float32)).cuda()



    def forward(self, input_loc_ : torch.Tensor, input_demand_ : torch.Tensor, input_depot_ : torch.Tensor):
        self_init_embed_depot = self.self_init_embed_depot(input_depot_);  input_depot_ = None
        getitem = self_init_embed_depot[(slice(None, None, None), None, slice(None, None, None))];  self_init_embed_depot = None
        getitem_1 = input_demand_[(slice(None, None, None), slice(None, None, None), None)];  input_demand_ = None
        cat = torch.cat((input_loc_, getitem_1), -1);  input_loc_ = getitem_1 = None
        self_init_embed = self.self_init_embed(cat);  cat = None
        cat_1 = torch.cat((getitem, self_init_embed), 1);  getitem = self_init_embed = None
        contiguous = cat_1.contiguous()
        view = contiguous.view(-1, 128);  contiguous = None
        contiguous_1 = cat_1.contiguous()
        view_1 = contiguous_1.view(-1, 128);  contiguous_1 = None
        getattr_getattr_self_embedder_layers___0_____0___module_w_query = self.getattr_getattr_self_embedder_layers___0_____0___module_W_query
        matmul = torch.matmul(view_1, getattr_getattr_self_embedder_layers___0_____0___module_w_query);  view_1 = getattr_getattr_self_embedder_layers___0_____0___module_w_query = None
        view_2 = matmul.view((8, 1024, 21, -1));  matmul = None
        getattr_getattr_self_embedder_layers___0_____0___module_w_key = self.getattr_getattr_self_embedder_layers___0_____0___module_W_key
        matmul_1 = torch.matmul(view, getattr_getattr_self_embedder_layers___0_____0___module_w_key);  getattr_getattr_self_embedder_layers___0_____0___module_w_key = None
        view_3 = matmul_1.view((8, 1024, 21, -1));  matmul_1 = None
        getattr_getattr_self_embedder_layers___0_____0___module_w_val = self.getattr_getattr_self_embedder_layers___0_____0___module_W_val
        matmul_2 = torch.matmul(view, getattr_getattr_self_embedder_layers___0_____0___module_w_val);  view = getattr_getattr_self_embedder_layers___0_____0___module_w_val = None
        view_4 = matmul_2.view((8, 1024, 21, -1));  matmul_2 = None
        transpose = view_3.transpose(2, 3);  view_3 = None
        matmul_3 = torch.matmul(view_2, transpose);  view_2 = transpose = None
        mul = 0.25 * matmul_3;  matmul_3 = None
        softmax = torch.softmax(mul, dim = -1);  mul = None
        matmul_4 = torch.matmul(softmax, view_4);  softmax = view_4 = None
        permute = matmul_4.permute(1, 2, 0, 3);  matmul_4 = None
        contiguous_2 = permute.contiguous();  permute = None
        view_5 = contiguous_2.view(-1, 128);  contiguous_2 = None
        getattr_getattr_self_embedder_layers___0_____0___module_w_out = self.getattr_getattr_self_embedder_layers___0_____0___module_W_out
        view_6 = getattr_getattr_self_embedder_layers___0_____0___module_w_out.view(-1, 128);  getattr_getattr_self_embedder_layers___0_____0___module_w_out = None
        mm = torch.mm(view_5, view_6);  view_5 = view_6 = None
        view_7 = mm.view(1024, 21, 128);  mm = None
        add = cat_1 + view_7;  cat_1 = view_7 = None
        view_8 = add.view(-1, 128);  add = None
        getattr_getattr_self_embedder_layers___0_____1___normalizer = self.getattr_getattr_self_embedder_layers___0_____1___normalizer(view_8);  view_8 = None
        view_9 = getattr_getattr_self_embedder_layers___0_____1___normalizer.view(1024, 21, 128);  getattr_getattr_self_embedder_layers___0_____1___normalizer = None
        getattr_getattr_self_embedder_layers___0_____2___module_0 = self.getattr_getattr_self_embedder_layers___0_____2___module_0(view_9)
        getattr_getattr_self_embedder_layers___0_____2___module_1 = self.getattr_getattr_self_embedder_layers___0_____2___module_1(getattr_getattr_self_embedder_layers___0_____2___module_0);  getattr_getattr_self_embedder_layers___0_____2___module_0 = None
        getattr_getattr_self_embedder_layers___0_____2___module_2 = self.getattr_getattr_self_embedder_layers___0_____2___module_2(getattr_getattr_self_embedder_layers___0_____2___module_1);  getattr_getattr_self_embedder_layers___0_____2___module_1 = None
        add_1 = view_9 + getattr_getattr_self_embedder_layers___0_____2___module_2;  view_9 = getattr_getattr_self_embedder_layers___0_____2___module_2 = None
        view_10 = add_1.view(-1, 128);  add_1 = None
        getattr_getattr_self_embedder_layers___0_____3___normalizer = self.getattr_getattr_self_embedder_layers___0_____3___normalizer(view_10);  view_10 = None
        view_11 = getattr_getattr_self_embedder_layers___0_____3___normalizer.view(1024, 21, 128);  getattr_getattr_self_embedder_layers___0_____3___normalizer = None
        contiguous_3 = view_11.contiguous()
        view_12 = contiguous_3.view(-1, 128);  contiguous_3 = None
        contiguous_4 = view_11.contiguous()
        view_13 = contiguous_4.view(-1, 128);  contiguous_4 = None
        getattr_getattr_self_embedder_layers___1_____0___module_w_query = self.getattr_getattr_self_embedder_layers___1_____0___module_W_query
        matmul_5 = torch.matmul(view_13, getattr_getattr_self_embedder_layers___1_____0___module_w_query);  view_13 = getattr_getattr_self_embedder_layers___1_____0___module_w_query = None
        view_14 = matmul_5.view((8, 1024, 21, -1));  matmul_5 = None
        getattr_getattr_self_embedder_layers___1_____0___module_w_key = self.getattr_getattr_self_embedder_layers___1_____0___module_W_key
        matmul_6 = torch.matmul(view_12, getattr_getattr_self_embedder_layers___1_____0___module_w_key);  getattr_getattr_self_embedder_layers___1_____0___module_w_key = None
        view_15 = matmul_6.view((8, 1024, 21, -1));  matmul_6 = None
        getattr_getattr_self_embedder_layers___1_____0___module_w_val = self.getattr_getattr_self_embedder_layers___1_____0___module_W_val
        matmul_7 = torch.matmul(view_12, getattr_getattr_self_embedder_layers___1_____0___module_w_val);  view_12 = getattr_getattr_self_embedder_layers___1_____0___module_w_val = None
        view_16 = matmul_7.view((8, 1024, 21, -1));  matmul_7 = None
        transpose_1 = view_15.transpose(2, 3);  view_15 = None
        matmul_8 = torch.matmul(view_14, transpose_1);  view_14 = transpose_1 = None
        mul_1 = 0.25 * matmul_8;  matmul_8 = None
        softmax_1 = torch.softmax(mul_1, dim = -1);  mul_1 = None
        matmul_9 = torch.matmul(softmax_1, view_16);  softmax_1 = view_16 = None
        permute_1 = matmul_9.permute(1, 2, 0, 3);  matmul_9 = None
        contiguous_5 = permute_1.contiguous();  permute_1 = None
        view_17 = contiguous_5.view(-1, 128);  contiguous_5 = None
        getattr_getattr_self_embedder_layers___1_____0___module_w_out = self.getattr_getattr_self_embedder_layers___1_____0___module_W_out
        view_18 = getattr_getattr_self_embedder_layers___1_____0___module_w_out.view(-1, 128);  getattr_getattr_self_embedder_layers___1_____0___module_w_out = None
        mm_1 = torch.mm(view_17, view_18);  view_17 = view_18 = None
        view_19 = mm_1.view(1024, 21, 128);  mm_1 = None
        add_2 = view_11 + view_19;  view_11 = view_19 = None
        view_20 = add_2.view(-1, 128);  add_2 = None
        getattr_getattr_self_embedder_layers___1_____1___normalizer = self.getattr_getattr_self_embedder_layers___1_____1___normalizer(view_20);  view_20 = None
        view_21 = getattr_getattr_self_embedder_layers___1_____1___normalizer.view(1024, 21, 128);  getattr_getattr_self_embedder_layers___1_____1___normalizer = None
        getattr_getattr_self_embedder_layers___1_____2___module_0 = self.getattr_getattr_self_embedder_layers___1_____2___module_0(view_21)
        getattr_getattr_self_embedder_layers___1_____2___module_1 = self.getattr_getattr_self_embedder_layers___1_____2___module_1(getattr_getattr_self_embedder_layers___1_____2___module_0);  getattr_getattr_self_embedder_layers___1_____2___module_0 = None
        getattr_getattr_self_embedder_layers___1_____2___module_2 = self.getattr_getattr_self_embedder_layers___1_____2___module_2(getattr_getattr_self_embedder_layers___1_____2___module_1);  getattr_getattr_self_embedder_layers___1_____2___module_1 = None
        add_3 = view_21 + getattr_getattr_self_embedder_layers___1_____2___module_2;  view_21 = getattr_getattr_self_embedder_layers___1_____2___module_2 = None
        view_22 = add_3.view(-1, 128);  add_3 = None
        getattr_getattr_self_embedder_layers___1_____3___normalizer = self.getattr_getattr_self_embedder_layers___1_____3___normalizer(view_22);  view_22 = None
        view_23 = getattr_getattr_self_embedder_layers___1_____3___normalizer.view(1024, 21, 128);  getattr_getattr_self_embedder_layers___1_____3___normalizer = None
        contiguous_6 = view_23.contiguous()
        view_24 = contiguous_6.view(-1, 128);  contiguous_6 = None
        contiguous_7 = view_23.contiguous()
        view_25 = contiguous_7.view(-1, 128);  contiguous_7 = None
        getattr_getattr_self_embedder_layers___2_____0___module_w_query = self.getattr_getattr_self_embedder_layers___2_____0___module_W_query
        matmul_10 = torch.matmul(view_25, getattr_getattr_self_embedder_layers___2_____0___module_w_query);  view_25 = getattr_getattr_self_embedder_layers___2_____0___module_w_query = None
        view_26 = matmul_10.view((8, 1024, 21, -1));  matmul_10 = None
        getattr_getattr_self_embedder_layers___2_____0___module_w_key = self.getattr_getattr_self_embedder_layers___2_____0___module_W_key
        matmul_11 = torch.matmul(view_24, getattr_getattr_self_embedder_layers___2_____0___module_w_key);  getattr_getattr_self_embedder_layers___2_____0___module_w_key = None
        view_27 = matmul_11.view((8, 1024, 21, -1));  matmul_11 = None
        getattr_getattr_self_embedder_layers___2_____0___module_w_val = self.getattr_getattr_self_embedder_layers___2_____0___module_W_val
        matmul_12 = torch.matmul(view_24, getattr_getattr_self_embedder_layers___2_____0___module_w_val);  view_24 = getattr_getattr_self_embedder_layers___2_____0___module_w_val = None
        view_28 = matmul_12.view((8, 1024, 21, -1));  matmul_12 = None
        transpose_2 = view_27.transpose(2, 3);  view_27 = None
        matmul_13 = torch.matmul(view_26, transpose_2);  view_26 = transpose_2 = None
        mul_2 = 0.25 * matmul_13;  matmul_13 = None
        softmax_2 = torch.softmax(mul_2, dim = -1);  mul_2 = None
        matmul_14 = torch.matmul(softmax_2, view_28);  softmax_2 = view_28 = None
        permute_2 = matmul_14.permute(1, 2, 0, 3);  matmul_14 = None
        contiguous_8 = permute_2.contiguous();  permute_2 = None
        view_29 = contiguous_8.view(-1, 128);  contiguous_8 = None
        getattr_getattr_self_embedder_layers___2_____0___module_w_out = self.getattr_getattr_self_embedder_layers___2_____0___module_W_out
        view_30 = getattr_getattr_self_embedder_layers___2_____0___module_w_out.view(-1, 128);  getattr_getattr_self_embedder_layers___2_____0___module_w_out = None
        mm_2 = torch.mm(view_29, view_30);  view_29 = view_30 = None
        view_31 = mm_2.view(1024, 21, 128);  mm_2 = None
        add_4 = view_23 + view_31;  view_23 = view_31 = None
        view_32 = add_4.view(-1, 128);  add_4 = None
        getattr_getattr_self_embedder_layers___2_____1___normalizer = self.getattr_getattr_self_embedder_layers___2_____1___normalizer(view_32);  view_32 = None
        view_33 = getattr_getattr_self_embedder_layers___2_____1___normalizer.view(1024, 21, 128);  getattr_getattr_self_embedder_layers___2_____1___normalizer = None
        getattr_getattr_self_embedder_layers___2_____2___module_0 = self.getattr_getattr_self_embedder_layers___2_____2___module_0(view_33)
        getattr_getattr_self_embedder_layers___2_____2___module_1 = self.getattr_getattr_self_embedder_layers___2_____2___module_1(getattr_getattr_self_embedder_layers___2_____2___module_0);  getattr_getattr_self_embedder_layers___2_____2___module_0 = None
        getattr_getattr_self_embedder_layers___2_____2___module_2 = self.getattr_getattr_self_embedder_layers___2_____2___module_2(getattr_getattr_self_embedder_layers___2_____2___module_1);  getattr_getattr_self_embedder_layers___2_____2___module_1 = None
        add_5 = view_33 + getattr_getattr_self_embedder_layers___2_____2___module_2;  view_33 = getattr_getattr_self_embedder_layers___2_____2___module_2 = None
        view_34 = add_5.view(-1, 128);  add_5 = None
        getattr_getattr_self_embedder_layers___2_____3___normalizer = self.getattr_getattr_self_embedder_layers___2_____3___normalizer(view_34);  view_34 = None
        view_35 = getattr_getattr_self_embedder_layers___2_____3___normalizer.view(1024, 21, 128);  getattr_getattr_self_embedder_layers___2_____3___normalizer = None
        mean = view_35.mean(dim = 1)
        return (view_35,)


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