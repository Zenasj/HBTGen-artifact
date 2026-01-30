import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, data):
        data = data.to('cuda')
        return self.linear(data)


if __name__ == '__main__':
    data = torch.zeros(4, 4)
    model = Model().cuda().eval()
    model = torch.compile(model)
    with torch.no_grad():
        out = model(data)

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
torch._dynamo.config.load_config(b'\x80\x02}q\x00(X\x0b\x00\x00\x00output_codeq\x01\x89X\r\x00\x00\x00log_file_nameq\x02NX\x07\x00\x00\x00verboseq\x03\x89X\x11\x00\x00\x00output_graph_codeq\x04\x89X\x12\x00\x00\x00verify_correctnessq\x05\x89X\x12\x00\x00\x00minimum_call_countq\x06K\x01X\x15\x00\x00\x00dead_code_eliminationq\x07\x88X\x10\x00\x00\x00cache_size_limitq\x08K@X\x14\x00\x00\x00specialize_int_floatq\t\x88X\x0e\x00\x00\x00dynamic_shapesq\n\x89X\x10\x00\x00\x00guard_nn_modulesq\x0b\x89X\x1b\x00\x00\x00traceable_tensor_subclassesq\x0cc__builtin__\nset\nq\r]q\x0e\x85q\x0fRq\x10X\x0f\x00\x00\x00suppress_errorsq\x11\x89X\x15\x00\x00\x00replay_record_enabledq\x12\x89X \x00\x00\x00rewrite_assert_with_torch_assertq\x13\x88X\x12\x00\x00\x00print_graph_breaksq\x14\x89X\x07\x00\x00\x00disableq\x15\x89X*\x00\x00\x00allowed_functions_module_string_ignorelistq\x16h\r]q\x17(X\x0b\x00\x00\x00torch._refsq\x18X\r\x00\x00\x00torch.testingq\x19X\r\x00\x00\x00torch._decompq\x1aX\x0c\x00\x00\x00torch._primsq\x1bX\x13\x00\x00\x00torch.distributionsq\x1ce\x85q\x1dRq\x1eX\x12\x00\x00\x00repro_forward_onlyq\x1f\x89X\x0f\x00\x00\x00repro_toleranceq G?PbM\xd2\xf1\xa9\xfcX\x16\x00\x00\x00capture_scalar_outputsq!\x89X\x19\x00\x00\x00enforce_cond_guards_matchq"\x88X\x0c\x00\x00\x00optimize_ddpq#\x88X\x1a\x00\x00\x00raise_on_ctx_manager_usageq$\x88X\x1c\x00\x00\x00raise_on_unsafe_aot_autogradq%\x89X\x17\x00\x00\x00raise_on_backend_changeq&\x89X\x18\x00\x00\x00error_on_nested_fx_traceq\'\x88X\t\x00\x00\x00allow_rnnq(\x89X\x08\x00\x00\x00base_dirq)XD\x00\x00\x00/home/PJLAB/zhaoqian/miniconda3/envs/pt2/lib/python3.8/site-packagesq*X\x0e\x00\x00\x00debug_dir_rootq+XD\x00\x00\x00/home/PJLAB/zhaoqian/projects/OpenMMLab/mmengine/torch_compile_debugq,X)\x00\x00\x00DO_NOT_USE_legacy_non_fake_example_inputsq-\x89X\x13\x00\x00\x00_save_config_ignoreq.h\r]q/(X\x0b\x00\x00\x00repro_afterq0X\x12\x00\x00\x00constant_functionsq1X!\x00\x00\x00skipfiles_inline_module_allowlistq2X\x0b\x00\x00\x00repro_levelq3e\x85q4Rq5u.')
torch._inductor.config.load_config(b'\x80\x02}q\x00(X\x05\x00\x00\x00debugq\x01\x89X\x10\x00\x00\x00disable_progressq\x02\x88X\x10\x00\x00\x00verbose_progressq\x03\x89X\x0b\x00\x00\x00cpp_wrapperq\x04\x89X\x03\x00\x00\x00dceq\x05\x89X\x14\x00\x00\x00static_weight_shapesq\x06\x88X\x0c\x00\x00\x00size_assertsq\x07\x88X\x10\x00\x00\x00pick_loop_ordersq\x08\x88X\x0f\x00\x00\x00inplace_buffersq\t\x88X\x11\x00\x00\x00benchmark_harnessq\n\x88X\x0f\x00\x00\x00epilogue_fusionq\x0b\x89X\x15\x00\x00\x00epilogue_fusion_firstq\x0c\x89X\x0f\x00\x00\x00pattern_matcherq\r\x88X\n\x00\x00\x00reorderingq\x0e\x89X\x0c\x00\x00\x00max_autotuneq\x0f\x89X\x17\x00\x00\x00realize_reads_thresholdq\x10K\x04X\x17\x00\x00\x00realize_bytes_thresholdq\x11M\xd0\x07X\x1b\x00\x00\x00realize_acc_reads_thresholdq\x12K\x08X\x0f\x00\x00\x00fallback_randomq\x13\x89X\x12\x00\x00\x00implicit_fallbacksq\x14\x88X\x0b\x00\x00\x00tune_layoutq\x15\x89X\x11\x00\x00\x00aggressive_fusionq\x16\x89X\x0f\x00\x00\x00max_fusion_sizeq\x17K@X\x1b\x00\x00\x00unroll_reductions_thresholdq\x18K\x08X\x0e\x00\x00\x00comment_originq\x19\x89X\x12\x00\x00\x00developer_warningsq\x1a\x88X\x0f\x00\x00\x00compile_threadsq\x1bK\x10X\x13\x00\x00\x00kernel_name_max_opsq\x1cK\nX\r\x00\x00\x00shape_paddingq\x1d\x89X\x0e\x00\x00\x00permute_fusionq\x1e\x89X\x1a\x00\x00\x00profiler_mark_wrapper_callq\x1f\x89X\x18\x00\x00\x00_raise_error_for_testingq \x89X\x0b\x00\x00\x00cpp.threadsq!J\xff\xff\xff\xffX\x13\x00\x00\x00cpp.dynamic_threadsq"\x89X\x0b\x00\x00\x00cpp.simdlenq#NX\x12\x00\x00\x00cpp.min_chunk_sizeq$M\x00\x10X\x07\x00\x00\x00cpp.cxxq%NX\x03\x00\x00\x00g++q&\x86q\'X\x19\x00\x00\x00cpp.enable_kernel_profileq(\x89X\x12\x00\x00\x00cpp.weight_prepackq)\x88X\x11\x00\x00\x00triton.cudagraphsq*\x89X\x17\x00\x00\x00triton.debug_sync_graphq+\x89X\x18\x00\x00\x00triton.debug_sync_kernelq,\x89X\x15\x00\x00\x00triton.dense_indexingq-\x89X\x10\x00\x00\x00triton.max_tilesq.K\x02X\x19\x00\x00\x00triton.autotune_pointwiseq/\x88X\'\x00\x00\x00triton.tiling_prevents_pointwise_fusionq0\x88X\'\x00\x00\x00triton.tiling_prevents_reduction_fusionq1\x88X\x1b\x00\x00\x00triton.ordered_kernel_namesq2\x89X\x1f\x00\x00\x00triton.descriptive_kernel_namesq3\x89X\x1c\x00\x00\x00triton.persistent_reductionsq4\x89X\r\x00\x00\x00trace.enabledq5\x89X\x0f\x00\x00\x00trace.debug_logq6\x88X\x0e\x00\x00\x00trace.info_logq7\x89X\x0e\x00\x00\x00trace.fx_graphq8\x88X\x1a\x00\x00\x00trace.fx_graph_transformedq9\x88X\x13\x00\x00\x00trace.ir_pre_fusionq:\x88X\x14\x00\x00\x00trace.ir_post_fusionq;\x88X\x11\x00\x00\x00trace.output_codeq<\x88X\x13\x00\x00\x00trace.graph_diagramq=\x89X\x15\x00\x00\x00trace.compile_profileq>\x89X\x10\x00\x00\x00trace.upload_tarq?Nu.')
torch._functorch.config.load_config(b'\x80\x02}q\x00(X\x11\x00\x00\x00use_functionalizeq\x01\x88X\x0f\x00\x00\x00use_fake_tensorq\x02\x88X\x16\x00\x00\x00fake_tensor_allow_metaq\x03\x88X\x0c\x00\x00\x00debug_assertq\x04\x88X\x14\x00\x00\x00debug_fake_cross_refq\x05\x89X\x11\x00\x00\x00debug_partitionerq\x06\x89X\x0c\x00\x00\x00debug_graphsq\x07\x89X\x0b\x00\x00\x00debug_jointq\x08\x89X\x12\x00\x00\x00use_dynamic_shapesq\t\x89X\x14\x00\x00\x00static_weight_shapesq\n\x88X\x03\x00\x00\x00cseq\x0b\x88X\x10\x00\x00\x00max_dist_from_bwq\x0cK\x03X\t\x00\x00\x00log_levelq\rK\x14u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


args = [((32, 3, 32, 32), (3072, 1024, 32, 1), torch.float32, 'cpu', False), ((32,), (1,), torch.int64, 'cpu', False)]
args = [rand_strided(sh, st, dt, dev).requires_grad_(rg) for (sh, st, dt, dev, rg) in args]


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_resnet_conv1 = Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).cuda()
        self.self_resnet_bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_relu = ReLU(inplace=True)
        self.self_resnet_maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.self_resnet_layer1_0_conv1 = Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_0_bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_0_relu = ReLU(inplace=True)
        self.self_resnet_layer1_0_conv2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_0_bn2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_0_conv3 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_0_bn3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_0_downsample_0 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_0_downsample_1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_1_conv1 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_1_bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_1_relu = ReLU(inplace=True)
        self.self_resnet_layer1_1_conv2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_1_bn2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_1_conv3 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_1_bn3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_2_conv1 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_2_bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_2_relu = ReLU(inplace=True)
        self.self_resnet_layer1_2_conv2 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_2_bn2 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer1_2_conv3 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer1_2_bn3 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_0_conv1 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_0_bn1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_0_relu = ReLU(inplace=True)
        self.self_resnet_layer2_0_conv2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_0_bn2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_0_conv3 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_0_bn3 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_0_downsample_0 = Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        self.self_resnet_layer2_0_downsample_1 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_1_conv1 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_1_bn1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_1_relu = ReLU(inplace=True)
        self.self_resnet_layer2_1_conv2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_1_bn2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_1_conv3 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_1_bn3 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_2_conv1 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_2_bn1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_2_relu = ReLU(inplace=True)
        self.self_resnet_layer2_2_conv2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_2_bn2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_2_conv3 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_2_bn3 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_3_conv1 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_3_bn1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_3_relu = ReLU(inplace=True)
        self.self_resnet_layer2_3_conv2 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_3_bn2 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer2_3_conv3 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer2_3_bn3 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_0_conv1 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_0_bn1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_0_relu = ReLU(inplace=True)
        self.self_resnet_layer3_0_conv2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_0_bn2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_0_conv3 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_0_bn3 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_0_downsample_0 = Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        self.self_resnet_layer3_0_downsample_1 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_1_conv1 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_1_bn1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_1_relu = ReLU(inplace=True)
        self.self_resnet_layer3_1_conv2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_1_bn2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_1_conv3 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_1_bn3 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_2_conv1 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_2_bn1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_2_relu = ReLU(inplace=True)
        self.self_resnet_layer3_2_conv2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_2_bn2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_2_conv3 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_2_bn3 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_3_conv1 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_3_bn1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_3_relu = ReLU(inplace=True)
        self.self_resnet_layer3_3_conv2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_3_bn2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_3_conv3 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_3_bn3 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_4_conv1 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_4_bn1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_4_relu = ReLU(inplace=True)
        self.self_resnet_layer3_4_conv2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_4_bn2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_4_conv3 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_4_bn3 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_5_conv1 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_5_bn1 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_5_relu = ReLU(inplace=True)
        self.self_resnet_layer3_5_conv2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_5_bn2 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer3_5_conv3 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer3_5_bn3 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_0_conv1 = Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_0_bn1 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_0_relu = ReLU(inplace=True)
        self.self_resnet_layer4_0_conv2 = Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_0_bn2 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_0_conv3 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_0_bn3 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_0_downsample_0 = Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False).cuda()
        self.self_resnet_layer4_0_downsample_1 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_1_conv1 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_1_bn1 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_1_relu = ReLU(inplace=True)
        self.self_resnet_layer4_1_conv2 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_1_bn2 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_1_conv3 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_1_bn3 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_2_conv1 = Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_2_bn1 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_2_relu = ReLU(inplace=True)
        self.self_resnet_layer4_2_conv2 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_2_bn2 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_layer4_2_conv3 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        self.self_resnet_layer4_2_bn3 = BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).cuda()
        self.self_resnet_avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.self_resnet_fc = Linear(in_features=2048, out_features=1000, bias=True).cuda()



    def forward(self, data_0_ : torch.Tensor, data_1_ : torch.Tensor):
        to = data_0_.to(device(type='cuda'), non_blocking = False);  data_0_ = None
        to_1 = data_1_.to(device(type='cuda'), non_blocking = False);  data_1_ = None
        self_resnet_conv1 = self.self_resnet_conv1(to);  to = None
        self_resnet_bn1 = self.self_resnet_bn1(self_resnet_conv1);  self_resnet_conv1 = None
        self_resnet_relu = self.self_resnet_relu(self_resnet_bn1);  self_resnet_bn1 = None
        self_resnet_maxpool = self.self_resnet_maxpool(self_resnet_relu);  self_resnet_relu = None
        self_resnet_layer1_0_conv1 = self.self_resnet_layer1_0_conv1(self_resnet_maxpool)
        self_resnet_layer1_0_bn1 = self.self_resnet_layer1_0_bn1(self_resnet_layer1_0_conv1);  self_resnet_layer1_0_conv1 = None
        self_resnet_layer1_0_relu = self.self_resnet_layer1_0_relu(self_resnet_layer1_0_bn1);  self_resnet_layer1_0_bn1 = None
        self_resnet_layer1_0_conv2 = self.self_resnet_layer1_0_conv2(self_resnet_layer1_0_relu);  self_resnet_layer1_0_relu = None
        self_resnet_layer1_0_bn2 = self.self_resnet_layer1_0_bn2(self_resnet_layer1_0_conv2);  self_resnet_layer1_0_conv2 = None
        self_resnet_layer1_0_relu_1 = self.self_resnet_layer1_0_relu(self_resnet_layer1_0_bn2);  self_resnet_layer1_0_bn2 = None
        self_resnet_layer1_0_conv3 = self.self_resnet_layer1_0_conv3(self_resnet_layer1_0_relu_1);  self_resnet_layer1_0_relu_1 = None
        self_resnet_layer1_0_bn3 = self.self_resnet_layer1_0_bn3(self_resnet_layer1_0_conv3);  self_resnet_layer1_0_conv3 = None
        self_resnet_layer1_0_downsample_0 = self.self_resnet_layer1_0_downsample_0(self_resnet_maxpool);  self_resnet_maxpool = None
        self_resnet_layer1_0_downsample_1 = self.self_resnet_layer1_0_downsample_1(self_resnet_layer1_0_downsample_0);  self_resnet_layer1_0_downsample_0 = None
        self_resnet_layer1_0_bn3 += self_resnet_layer1_0_downsample_1;  iadd = self_resnet_layer1_0_bn3;  self_resnet_layer1_0_bn3 = self_resnet_layer1_0_downsample_1 = None
        self_resnet_layer1_0_relu_2 = self.self_resnet_layer1_0_relu(iadd);  iadd = None
        self_resnet_layer1_1_conv1 = self.self_resnet_layer1_1_conv1(self_resnet_layer1_0_relu_2)
        self_resnet_layer1_1_bn1 = self.self_resnet_layer1_1_bn1(self_resnet_layer1_1_conv1);  self_resnet_layer1_1_conv1 = None
        self_resnet_layer1_1_relu = self.self_resnet_layer1_1_relu(self_resnet_layer1_1_bn1);  self_resnet_layer1_1_bn1 = None
        self_resnet_layer1_1_conv2 = self.self_resnet_layer1_1_conv2(self_resnet_layer1_1_relu);  self_resnet_layer1_1_relu = None
        self_resnet_layer1_1_bn2 = self.self_resnet_layer1_1_bn2(self_resnet_layer1_1_conv2);  self_resnet_layer1_1_conv2 = None
        self_resnet_layer1_1_relu_1 = self.self_resnet_layer1_1_relu(self_resnet_layer1_1_bn2);  self_resnet_layer1_1_bn2 = None
        self_resnet_layer1_1_conv3 = self.self_resnet_layer1_1_conv3(self_resnet_layer1_1_relu_1);  self_resnet_layer1_1_relu_1 = None
        self_resnet_layer1_1_bn3 = self.self_resnet_layer1_1_bn3(self_resnet_layer1_1_conv3);  self_resnet_layer1_1_conv3 = None
        self_resnet_layer1_1_bn3 += self_resnet_layer1_0_relu_2;  iadd_1 = self_resnet_layer1_1_bn3;  self_resnet_layer1_1_bn3 = self_resnet_layer1_0_relu_2 = None
        self_resnet_layer1_1_relu_2 = self.self_resnet_layer1_1_relu(iadd_1);  iadd_1 = None
        self_resnet_layer1_2_conv1 = self.self_resnet_layer1_2_conv1(self_resnet_layer1_1_relu_2)
        self_resnet_layer1_2_bn1 = self.self_resnet_layer1_2_bn1(self_resnet_layer1_2_conv1);  self_resnet_layer1_2_conv1 = None
        self_resnet_layer1_2_relu = self.self_resnet_layer1_2_relu(self_resnet_layer1_2_bn1);  self_resnet_layer1_2_bn1 = None
        self_resnet_layer1_2_conv2 = self.self_resnet_layer1_2_conv2(self_resnet_layer1_2_relu);  self_resnet_layer1_2_relu = None
        self_resnet_layer1_2_bn2 = self.self_resnet_layer1_2_bn2(self_resnet_layer1_2_conv2);  self_resnet_layer1_2_conv2 = None
        self_resnet_layer1_2_relu_1 = self.self_resnet_layer1_2_relu(self_resnet_layer1_2_bn2);  self_resnet_layer1_2_bn2 = None
        self_resnet_layer1_2_conv3 = self.self_resnet_layer1_2_conv3(self_resnet_layer1_2_relu_1);  self_resnet_layer1_2_relu_1 = None
        self_resnet_layer1_2_bn3 = self.self_resnet_layer1_2_bn3(self_resnet_layer1_2_conv3);  self_resnet_layer1_2_conv3 = None
        self_resnet_layer1_2_bn3 += self_resnet_layer1_1_relu_2;  iadd_2 = self_resnet_layer1_2_bn3;  self_resnet_layer1_2_bn3 = self_resnet_layer1_1_relu_2 = None
        self_resnet_layer1_2_relu_2 = self.self_resnet_layer1_2_relu(iadd_2);  iadd_2 = None
        self_resnet_layer2_0_conv1 = self.self_resnet_layer2_0_conv1(self_resnet_layer1_2_relu_2)
        self_resnet_layer2_0_bn1 = self.self_resnet_layer2_0_bn1(self_resnet_layer2_0_conv1);  self_resnet_layer2_0_conv1 = None
        self_resnet_layer2_0_relu = self.self_resnet_layer2_0_relu(self_resnet_layer2_0_bn1);  self_resnet_layer2_0_bn1 = None
        self_resnet_layer2_0_conv2 = self.self_resnet_layer2_0_conv2(self_resnet_layer2_0_relu);  self_resnet_layer2_0_relu = None
        self_resnet_layer2_0_bn2 = self.self_resnet_layer2_0_bn2(self_resnet_layer2_0_conv2);  self_resnet_layer2_0_conv2 = None
        self_resnet_layer2_0_relu_1 = self.self_resnet_layer2_0_relu(self_resnet_layer2_0_bn2);  self_resnet_layer2_0_bn2 = None
        self_resnet_layer2_0_conv3 = self.self_resnet_layer2_0_conv3(self_resnet_layer2_0_relu_1);  self_resnet_layer2_0_relu_1 = None
        self_resnet_layer2_0_bn3 = self.self_resnet_layer2_0_bn3(self_resnet_layer2_0_conv3);  self_resnet_layer2_0_conv3 = None
        self_resnet_layer2_0_downsample_0 = self.self_resnet_layer2_0_downsample_0(self_resnet_layer1_2_relu_2);  self_resnet_layer1_2_relu_2 = None
        self_resnet_layer2_0_downsample_1 = self.self_resnet_layer2_0_downsample_1(self_resnet_layer2_0_downsample_0);  self_resnet_layer2_0_downsample_0 = None
        self_resnet_layer2_0_bn3 += self_resnet_layer2_0_downsample_1;  iadd_3 = self_resnet_layer2_0_bn3;  self_resnet_layer2_0_bn3 = self_resnet_layer2_0_downsample_1 = None
        self_resnet_layer2_0_relu_2 = self.self_resnet_layer2_0_relu(iadd_3);  iadd_3 = None
        self_resnet_layer2_1_conv1 = self.self_resnet_layer2_1_conv1(self_resnet_layer2_0_relu_2)
        self_resnet_layer2_1_bn1 = self.self_resnet_layer2_1_bn1(self_resnet_layer2_1_conv1);  self_resnet_layer2_1_conv1 = None
        self_resnet_layer2_1_relu = self.self_resnet_layer2_1_relu(self_resnet_layer2_1_bn1);  self_resnet_layer2_1_bn1 = None
        self_resnet_layer2_1_conv2 = self.self_resnet_layer2_1_conv2(self_resnet_layer2_1_relu);  self_resnet_layer2_1_relu = None
        self_resnet_layer2_1_bn2 = self.self_resnet_layer2_1_bn2(self_resnet_layer2_1_conv2);  self_resnet_layer2_1_conv2 = None
        self_resnet_layer2_1_relu_1 = self.self_resnet_layer2_1_relu(self_resnet_layer2_1_bn2);  self_resnet_layer2_1_bn2 = None
        self_resnet_layer2_1_conv3 = self.self_resnet_layer2_1_conv3(self_resnet_layer2_1_relu_1);  self_resnet_layer2_1_relu_1 = None
        self_resnet_layer2_1_bn3 = self.self_resnet_layer2_1_bn3(self_resnet_layer2_1_conv3);  self_resnet_layer2_1_conv3 = None
        self_resnet_layer2_1_bn3 += self_resnet_layer2_0_relu_2;  iadd_4 = self_resnet_layer2_1_bn3;  self_resnet_layer2_1_bn3 = self_resnet_layer2_0_relu_2 = None
        self_resnet_layer2_1_relu_2 = self.self_resnet_layer2_1_relu(iadd_4);  iadd_4 = None
        self_resnet_layer2_2_conv1 = self.self_resnet_layer2_2_conv1(self_resnet_layer2_1_relu_2)
        self_resnet_layer2_2_bn1 = self.self_resnet_layer2_2_bn1(self_resnet_layer2_2_conv1);  self_resnet_layer2_2_conv1 = None
        self_resnet_layer2_2_relu = self.self_resnet_layer2_2_relu(self_resnet_layer2_2_bn1);  self_resnet_layer2_2_bn1 = None
        self_resnet_layer2_2_conv2 = self.self_resnet_layer2_2_conv2(self_resnet_layer2_2_relu);  self_resnet_layer2_2_relu = None
        self_resnet_layer2_2_bn2 = self.self_resnet_layer2_2_bn2(self_resnet_layer2_2_conv2);  self_resnet_layer2_2_conv2 = None
        self_resnet_layer2_2_relu_1 = self.self_resnet_layer2_2_relu(self_resnet_layer2_2_bn2);  self_resnet_layer2_2_bn2 = None
        self_resnet_layer2_2_conv3 = self.self_resnet_layer2_2_conv3(self_resnet_layer2_2_relu_1);  self_resnet_layer2_2_relu_1 = None
        self_resnet_layer2_2_bn3 = self.self_resnet_layer2_2_bn3(self_resnet_layer2_2_conv3);  self_resnet_layer2_2_conv3 = None
        self_resnet_layer2_2_bn3 += self_resnet_layer2_1_relu_2;  iadd_5 = self_resnet_layer2_2_bn3;  self_resnet_layer2_2_bn3 = self_resnet_layer2_1_relu_2 = None
        self_resnet_layer2_2_relu_2 = self.self_resnet_layer2_2_relu(iadd_5);  iadd_5 = None
        self_resnet_layer2_3_conv1 = self.self_resnet_layer2_3_conv1(self_resnet_layer2_2_relu_2)
        self_resnet_layer2_3_bn1 = self.self_resnet_layer2_3_bn1(self_resnet_layer2_3_conv1);  self_resnet_layer2_3_conv1 = None
        self_resnet_layer2_3_relu = self.self_resnet_layer2_3_relu(self_resnet_layer2_3_bn1);  self_resnet_layer2_3_bn1 = None
        self_resnet_layer2_3_conv2 = self.self_resnet_layer2_3_conv2(self_resnet_layer2_3_relu);  self_resnet_layer2_3_relu = None
        self_resnet_layer2_3_bn2 = self.self_resnet_layer2_3_bn2(self_resnet_layer2_3_conv2);  self_resnet_layer2_3_conv2 = None
        self_resnet_layer2_3_relu_1 = self.self_resnet_layer2_3_relu(self_resnet_layer2_3_bn2);  self_resnet_layer2_3_bn2 = None
        self_resnet_layer2_3_conv3 = self.self_resnet_layer2_3_conv3(self_resnet_layer2_3_relu_1);  self_resnet_layer2_3_relu_1 = None
        self_resnet_layer2_3_bn3 = self.self_resnet_layer2_3_bn3(self_resnet_layer2_3_conv3);  self_resnet_layer2_3_conv3 = None
        self_resnet_layer2_3_bn3 += self_resnet_layer2_2_relu_2;  iadd_6 = self_resnet_layer2_3_bn3;  self_resnet_layer2_3_bn3 = self_resnet_layer2_2_relu_2 = None
        self_resnet_layer2_3_relu_2 = self.self_resnet_layer2_3_relu(iadd_6);  iadd_6 = None
        self_resnet_layer3_0_conv1 = self.self_resnet_layer3_0_conv1(self_resnet_layer2_3_relu_2)
        self_resnet_layer3_0_bn1 = self.self_resnet_layer3_0_bn1(self_resnet_layer3_0_conv1);  self_resnet_layer3_0_conv1 = None
        self_resnet_layer3_0_relu = self.self_resnet_layer3_0_relu(self_resnet_layer3_0_bn1);  self_resnet_layer3_0_bn1 = None
        self_resnet_layer3_0_conv2 = self.self_resnet_layer3_0_conv2(self_resnet_layer3_0_relu);  self_resnet_layer3_0_relu = None
        self_resnet_layer3_0_bn2 = self.self_resnet_layer3_0_bn2(self_resnet_layer3_0_conv2);  self_resnet_layer3_0_conv2 = None
        self_resnet_layer3_0_relu_1 = self.self_resnet_layer3_0_relu(self_resnet_layer3_0_bn2);  self_resnet_layer3_0_bn2 = None
        self_resnet_layer3_0_conv3 = self.self_resnet_layer3_0_conv3(self_resnet_layer3_0_relu_1);  self_resnet_layer3_0_relu_1 = None
        self_resnet_layer3_0_bn3 = self.self_resnet_layer3_0_bn3(self_resnet_layer3_0_conv3);  self_resnet_layer3_0_conv3 = None
        self_resnet_layer3_0_downsample_0 = self.self_resnet_layer3_0_downsample_0(self_resnet_layer2_3_relu_2);  self_resnet_layer2_3_relu_2 = None
        self_resnet_layer3_0_downsample_1 = self.self_resnet_layer3_0_downsample_1(self_resnet_layer3_0_downsample_0);  self_resnet_layer3_0_downsample_0 = None
        self_resnet_layer3_0_bn3 += self_resnet_layer3_0_downsample_1;  iadd_7 = self_resnet_layer3_0_bn3;  self_resnet_layer3_0_bn3 = self_resnet_layer3_0_downsample_1 = None
        self_resnet_layer3_0_relu_2 = self.self_resnet_layer3_0_relu(iadd_7);  iadd_7 = None
        self_resnet_layer3_1_conv1 = self.self_resnet_layer3_1_conv1(self_resnet_layer3_0_relu_2)
        self_resnet_layer3_1_bn1 = self.self_resnet_layer3_1_bn1(self_resnet_layer3_1_conv1);  self_resnet_layer3_1_conv1 = None
        self_resnet_layer3_1_relu = self.self_resnet_layer3_1_relu(self_resnet_layer3_1_bn1);  self_resnet_layer3_1_bn1 = None
        self_resnet_layer3_1_conv2 = self.self_resnet_layer3_1_conv2(self_resnet_layer3_1_relu);  self_resnet_layer3_1_relu = None
        self_resnet_layer3_1_bn2 = self.self_resnet_layer3_1_bn2(self_resnet_layer3_1_conv2);  self_resnet_layer3_1_conv2 = None
        self_resnet_layer3_1_relu_1 = self.self_resnet_layer3_1_relu(self_resnet_layer3_1_bn2);  self_resnet_layer3_1_bn2 = None
        self_resnet_layer3_1_conv3 = self.self_resnet_layer3_1_conv3(self_resnet_layer3_1_relu_1);  self_resnet_layer3_1_relu_1 = None
        self_resnet_layer3_1_bn3 = self.self_resnet_layer3_1_bn3(self_resnet_layer3_1_conv3);  self_resnet_layer3_1_conv3 = None
        self_resnet_layer3_1_bn3 += self_resnet_layer3_0_relu_2;  iadd_8 = self_resnet_layer3_1_bn3;  self_resnet_layer3_1_bn3 = self_resnet_layer3_0_relu_2 = None
        self_resnet_layer3_1_relu_2 = self.self_resnet_layer3_1_relu(iadd_8);  iadd_8 = None
        self_resnet_layer3_2_conv1 = self.self_resnet_layer3_2_conv1(self_resnet_layer3_1_relu_2)
        self_resnet_layer3_2_bn1 = self.self_resnet_layer3_2_bn1(self_resnet_layer3_2_conv1);  self_resnet_layer3_2_conv1 = None
        self_resnet_layer3_2_relu = self.self_resnet_layer3_2_relu(self_resnet_layer3_2_bn1);  self_resnet_layer3_2_bn1 = None
        self_resnet_layer3_2_conv2 = self.self_resnet_layer3_2_conv2(self_resnet_layer3_2_relu);  self_resnet_layer3_2_relu = None
        self_resnet_layer3_2_bn2 = self.self_resnet_layer3_2_bn2(self_resnet_layer3_2_conv2);  self_resnet_layer3_2_conv2 = None
        self_resnet_layer3_2_relu_1 = self.self_resnet_layer3_2_relu(self_resnet_layer3_2_bn2);  self_resnet_layer3_2_bn2 = None
        self_resnet_layer3_2_conv3 = self.self_resnet_layer3_2_conv3(self_resnet_layer3_2_relu_1);  self_resnet_layer3_2_relu_1 = None
        self_resnet_layer3_2_bn3 = self.self_resnet_layer3_2_bn3(self_resnet_layer3_2_conv3);  self_resnet_layer3_2_conv3 = None
        self_resnet_layer3_2_bn3 += self_resnet_layer3_1_relu_2;  iadd_9 = self_resnet_layer3_2_bn3;  self_resnet_layer3_2_bn3 = self_resnet_layer3_1_relu_2 = None
        self_resnet_layer3_2_relu_2 = self.self_resnet_layer3_2_relu(iadd_9);  iadd_9 = None
        self_resnet_layer3_3_conv1 = self.self_resnet_layer3_3_conv1(self_resnet_layer3_2_relu_2)
        self_resnet_layer3_3_bn1 = self.self_resnet_layer3_3_bn1(self_resnet_layer3_3_conv1);  self_resnet_layer3_3_conv1 = None
        self_resnet_layer3_3_relu = self.self_resnet_layer3_3_relu(self_resnet_layer3_3_bn1);  self_resnet_layer3_3_bn1 = None
        self_resnet_layer3_3_conv2 = self.self_resnet_layer3_3_conv2(self_resnet_layer3_3_relu);  self_resnet_layer3_3_relu = None
        self_resnet_layer3_3_bn2 = self.self_resnet_layer3_3_bn2(self_resnet_layer3_3_conv2);  self_resnet_layer3_3_conv2 = None
        self_resnet_layer3_3_relu_1 = self.self_resnet_layer3_3_relu(self_resnet_layer3_3_bn2);  self_resnet_layer3_3_bn2 = None
        self_resnet_layer3_3_conv3 = self.self_resnet_layer3_3_conv3(self_resnet_layer3_3_relu_1);  self_resnet_layer3_3_relu_1 = None
        self_resnet_layer3_3_bn3 = self.self_resnet_layer3_3_bn3(self_resnet_layer3_3_conv3);  self_resnet_layer3_3_conv3 = None
        self_resnet_layer3_3_bn3 += self_resnet_layer3_2_relu_2;  iadd_10 = self_resnet_layer3_3_bn3;  self_resnet_layer3_3_bn3 = self_resnet_layer3_2_relu_2 = None
        self_resnet_layer3_3_relu_2 = self.self_resnet_layer3_3_relu(iadd_10);  iadd_10 = None
        self_resnet_layer3_4_conv1 = self.self_resnet_layer3_4_conv1(self_resnet_layer3_3_relu_2)
        self_resnet_layer3_4_bn1 = self.self_resnet_layer3_4_bn1(self_resnet_layer3_4_conv1);  self_resnet_layer3_4_conv1 = None
        self_resnet_layer3_4_relu = self.self_resnet_layer3_4_relu(self_resnet_layer3_4_bn1);  self_resnet_layer3_4_bn1 = None
        self_resnet_layer3_4_conv2 = self.self_resnet_layer3_4_conv2(self_resnet_layer3_4_relu);  self_resnet_layer3_4_relu = None
        self_resnet_layer3_4_bn2 = self.self_resnet_layer3_4_bn2(self_resnet_layer3_4_conv2);  self_resnet_layer3_4_conv2 = None
        self_resnet_layer3_4_relu_1 = self.self_resnet_layer3_4_relu(self_resnet_layer3_4_bn2);  self_resnet_layer3_4_bn2 = None
        self_resnet_layer3_4_conv3 = self.self_resnet_layer3_4_conv3(self_resnet_layer3_4_relu_1);  self_resnet_layer3_4_relu_1 = None
        self_resnet_layer3_4_bn3 = self.self_resnet_layer3_4_bn3(self_resnet_layer3_4_conv3);  self_resnet_layer3_4_conv3 = None
        self_resnet_layer3_4_bn3 += self_resnet_layer3_3_relu_2;  iadd_11 = self_resnet_layer3_4_bn3;  self_resnet_layer3_4_bn3 = self_resnet_layer3_3_relu_2 = None
        self_resnet_layer3_4_relu_2 = self.self_resnet_layer3_4_relu(iadd_11);  iadd_11 = None
        self_resnet_layer3_5_conv1 = self.self_resnet_layer3_5_conv1(self_resnet_layer3_4_relu_2)
        self_resnet_layer3_5_bn1 = self.self_resnet_layer3_5_bn1(self_resnet_layer3_5_conv1);  self_resnet_layer3_5_conv1 = None
        self_resnet_layer3_5_relu = self.self_resnet_layer3_5_relu(self_resnet_layer3_5_bn1);  self_resnet_layer3_5_bn1 = None
        self_resnet_layer3_5_conv2 = self.self_resnet_layer3_5_conv2(self_resnet_layer3_5_relu);  self_resnet_layer3_5_relu = None
        self_resnet_layer3_5_bn2 = self.self_resnet_layer3_5_bn2(self_resnet_layer3_5_conv2);  self_resnet_layer3_5_conv2 = None
        self_resnet_layer3_5_relu_1 = self.self_resnet_layer3_5_relu(self_resnet_layer3_5_bn2);  self_resnet_layer3_5_bn2 = None
        self_resnet_layer3_5_conv3 = self.self_resnet_layer3_5_conv3(self_resnet_layer3_5_relu_1);  self_resnet_layer3_5_relu_1 = None
        self_resnet_layer3_5_bn3 = self.self_resnet_layer3_5_bn3(self_resnet_layer3_5_conv3);  self_resnet_layer3_5_conv3 = None
        self_resnet_layer3_5_bn3 += self_resnet_layer3_4_relu_2;  iadd_12 = self_resnet_layer3_5_bn3;  self_resnet_layer3_5_bn3 = self_resnet_layer3_4_relu_2 = None
        self_resnet_layer3_5_relu_2 = self.self_resnet_layer3_5_relu(iadd_12);  iadd_12 = None
        self_resnet_layer4_0_conv1 = self.self_resnet_layer4_0_conv1(self_resnet_layer3_5_relu_2)
        self_resnet_layer4_0_bn1 = self.self_resnet_layer4_0_bn1(self_resnet_layer4_0_conv1);  self_resnet_layer4_0_conv1 = None
        self_resnet_layer4_0_relu = self.self_resnet_layer4_0_relu(self_resnet_layer4_0_bn1);  self_resnet_layer4_0_bn1 = None
        self_resnet_layer4_0_conv2 = self.self_resnet_layer4_0_conv2(self_resnet_layer4_0_relu);  self_resnet_layer4_0_relu = None
        self_resnet_layer4_0_bn2 = self.self_resnet_layer4_0_bn2(self_resnet_layer4_0_conv2);  self_resnet_layer4_0_conv2 = None
        self_resnet_layer4_0_relu_1 = self.self_resnet_layer4_0_relu(self_resnet_layer4_0_bn2);  self_resnet_layer4_0_bn2 = None
        self_resnet_layer4_0_conv3 = self.self_resnet_layer4_0_conv3(self_resnet_layer4_0_relu_1);  self_resnet_layer4_0_relu_1 = None
        self_resnet_layer4_0_bn3 = self.self_resnet_layer4_0_bn3(self_resnet_layer4_0_conv3);  self_resnet_layer4_0_conv3 = None
        self_resnet_layer4_0_downsample_0 = self.self_resnet_layer4_0_downsample_0(self_resnet_layer3_5_relu_2);  self_resnet_layer3_5_relu_2 = None
        self_resnet_layer4_0_downsample_1 = self.self_resnet_layer4_0_downsample_1(self_resnet_layer4_0_downsample_0);  self_resnet_layer4_0_downsample_0 = None
        self_resnet_layer4_0_bn3 += self_resnet_layer4_0_downsample_1;  iadd_13 = self_resnet_layer4_0_bn3;  self_resnet_layer4_0_bn3 = self_resnet_layer4_0_downsample_1 = None
        self_resnet_layer4_0_relu_2 = self.self_resnet_layer4_0_relu(iadd_13);  iadd_13 = None
        self_resnet_layer4_1_conv1 = self.self_resnet_layer4_1_conv1(self_resnet_layer4_0_relu_2)
        self_resnet_layer4_1_bn1 = self.self_resnet_layer4_1_bn1(self_resnet_layer4_1_conv1);  self_resnet_layer4_1_conv1 = None
        self_resnet_layer4_1_relu = self.self_resnet_layer4_1_relu(self_resnet_layer4_1_bn1);  self_resnet_layer4_1_bn1 = None
        self_resnet_layer4_1_conv2 = self.self_resnet_layer4_1_conv2(self_resnet_layer4_1_relu);  self_resnet_layer4_1_relu = None
        self_resnet_layer4_1_bn2 = self.self_resnet_layer4_1_bn2(self_resnet_layer4_1_conv2);  self_resnet_layer4_1_conv2 = None
        self_resnet_layer4_1_relu_1 = self.self_resnet_layer4_1_relu(self_resnet_layer4_1_bn2);  self_resnet_layer4_1_bn2 = None
        self_resnet_layer4_1_conv3 = self.self_resnet_layer4_1_conv3(self_resnet_layer4_1_relu_1);  self_resnet_layer4_1_relu_1 = None
        self_resnet_layer4_1_bn3 = self.self_resnet_layer4_1_bn3(self_resnet_layer4_1_conv3);  self_resnet_layer4_1_conv3 = None
        self_resnet_layer4_1_bn3 += self_resnet_layer4_0_relu_2;  iadd_14 = self_resnet_layer4_1_bn3;  self_resnet_layer4_1_bn3 = self_resnet_layer4_0_relu_2 = None
        self_resnet_layer4_1_relu_2 = self.self_resnet_layer4_1_relu(iadd_14);  iadd_14 = None
        self_resnet_layer4_2_conv1 = self.self_resnet_layer4_2_conv1(self_resnet_layer4_1_relu_2)
        self_resnet_layer4_2_bn1 = self.self_resnet_layer4_2_bn1(self_resnet_layer4_2_conv1);  self_resnet_layer4_2_conv1 = None
        self_resnet_layer4_2_relu = self.self_resnet_layer4_2_relu(self_resnet_layer4_2_bn1);  self_resnet_layer4_2_bn1 = None
        self_resnet_layer4_2_conv2 = self.self_resnet_layer4_2_conv2(self_resnet_layer4_2_relu);  self_resnet_layer4_2_relu = None
        self_resnet_layer4_2_bn2 = self.self_resnet_layer4_2_bn2(self_resnet_layer4_2_conv2);  self_resnet_layer4_2_conv2 = None
        self_resnet_layer4_2_relu_1 = self.self_resnet_layer4_2_relu(self_resnet_layer4_2_bn2);  self_resnet_layer4_2_bn2 = None
        self_resnet_layer4_2_conv3 = self.self_resnet_layer4_2_conv3(self_resnet_layer4_2_relu_1);  self_resnet_layer4_2_relu_1 = None
        self_resnet_layer4_2_bn3 = self.self_resnet_layer4_2_bn3(self_resnet_layer4_2_conv3);  self_resnet_layer4_2_conv3 = None
        self_resnet_layer4_2_bn3 += self_resnet_layer4_1_relu_2;  iadd_15 = self_resnet_layer4_2_bn3;  self_resnet_layer4_2_bn3 = self_resnet_layer4_1_relu_2 = None
        self_resnet_layer4_2_relu_2 = self.self_resnet_layer4_2_relu(iadd_15);  iadd_15 = None
        self_resnet_avgpool = self.self_resnet_avgpool(self_resnet_layer4_2_relu_2);  self_resnet_layer4_2_relu_2 = None
        flatten = torch.flatten(self_resnet_avgpool, 1);  self_resnet_avgpool = None
        self_resnet_fc = self.self_resnet_fc(flatten);  flatten = None
        return (self_resnet_fc, to_1)


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

data = data_preprocessor(data)
result = model(data)