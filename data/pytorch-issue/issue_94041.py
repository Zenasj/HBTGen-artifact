import torch.nn as nn

compile = dict(
    target='train_step',  # (train_step, forward, model)
    verbose=True,
    backend='inductor',  
    dynamic=False, 
)

isolate_fails_code_str = None


import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

import torch._dynamo.config
import torch._inductor.config
torch._dynamo.config.load_config(b'\x80\x04\x95\x12\x08\x00\x00\x00\x00\x00\x00}\x94(\x8c\x08__name__\x94\x8c\x14torch._dynamo.config\x94\x8c\x07__doc__\x94N\x8c\x0b__package__\x94\x8c\rtorch._dynamo\x94\x8c\n__loader__\x94\x8c\x1a_frozen_importlib_external\x94\x8c\x10SourceFileLoader\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94h\x02\x8c\x04path\x94\x8cb/mnt/petrelfs/xiexinchen/anaconda3/envs/pt2.0/lib/python3.10/site-packages/torch/_dynamo/config.py\x94ub\x8c\x08__spec__\x94\x8c\x11_frozen_importlib\x94\x8c\nModuleSpec\x94\x93\x94)\x81\x94}\x94(h\x0ch\x02\x8c\x06loader\x94h\n\x8c\x06origin\x94h\x0e\x8c\x0cloader_state\x94N\x8c\x1asubmodule_search_locations\x94N\x8c\r_set_fileattr\x94\x88\x8c\x07_cached\x94\x8c{/mnt/petrelfs/xiexinchen/anaconda3/envs/pt2.0/lib/python3.10/site-packages/torch/_dynamo/__pycache__/config.cpython-310.pyc\x94\x8c\r_initializing\x94\x89ub\x8c\x08__file__\x94h\x0e\x8c\n__cached__\x94h\x1b\x8c\x07abspath\x94\x8c\tposixpath\x94h\x1f\x93\x94\x8c\x07dirname\x94h h"\x93\x94\x8c\x0eHAS_REFS_PRIMS\x94\x88\x8c\tlog_level\x94K\x1e\x8c\x0boutput_code\x94\x89\x8c\rlog_file_name\x94N\x8c\x07verbose\x94\x88\x8c\x11output_graph_code\x94\x89\x8c\x12verify_correctness\x94\x89\x8c\x12minimum_call_count\x94K\x01\x8c\x15dead_code_elimination\x94\x88\x8c\x10cache_size_limit\x94K@\x8c\x14specialize_int_float\x94\x88\x8c\x0edynamic_shapes\x94\x89\x8c\x10guard_nn_modules\x94\x89\x8c\x0cnormalize_ir\x94\x89\x8c\x1btraceable_tensor_subclasses\x94\x8f\x94\x8c\x0fsuppress_errors\x94\x89\x8c\x15replay_record_enabled\x94\x89\x8c rewrite_assert_with_torch_assert\x94\x88\x8c\x12print_graph_breaks\x94\x89\x8c\x07disable\x94\x89\x8c*allowed_functions_module_string_ignorelist\x94\x8f\x94(\x8c\x0btorch._refs\x94\x8c\x13torch.distributions\x94\x8c\rtorch.testing\x94\x8c\x0ctorch._prims\x94\x8c\rtorch._decomp\x94\x90\x8c\x16capture_scalar_outputs\x94\x89\x8c\x19enforce_cond_guards_match\x94\x88\x8c\x0coptimize_ddp\x94\x88\x8c\x1araise_on_ctx_manager_usage\x94\x88\x8c\x1craise_on_unsafe_aot_autograd\x94\x89\x8c\rdynamo_import\x94\x8c\rtorch._dynamo\x94\x8c\x0finductor_import\x94\x8c\x0ftorch._inductor\x94\x8c\x18error_on_nested_fx_trace\x94\x88\x8c\tallow_rnn\x94\x89\x8c\x08base_dir\x94\x8cJ/mnt/petrelfs/xiexinchen/anaconda3/envs/pt2.0/lib/python3.10/site-packages\x94\x8c\x0edebug_dir_root\x94\x8c:/mnt/petrelfs/xiexinchen/mmsegv2-pt2.0/torch_compile_debug\x94\x8c)DO_NOT_USE_legacy_non_fake_example_inputs\x94\x89\x8c\x15_AccessLimitingConfig\x94}\x94(\x8c\n__module__\x94h\x02\x8c\x0b__setattr__\x94h\x02\x8c!_AccessLimitingConfig.__setattr__\x94\x93\x94h\x03Nu\x8c\x15_allowed_config_names\x94\x8f\x94(h8h2hOh\x0fh,\x8c\x0brepro_after\x94\x8c\x07logging\x94h"h5\x8c\x05torch\x94h0hM\x8c\x03sys\x94h(hJh)hEh*h1hD\x8c\nModuleType\x94hIh\x1fh+\x8c\x02os\x94h9h@\x8c\x12constant_functions\x94\x8c\x0c__builtins__\x94h-\x8c!skipfiles_inline_module_allowlist\x94h\x04h$hAh\x1eh%h4hBh&hP\x8c\x0eexternal_utils\x94hKh.h\'h\x06h6h\x03hCh\x01h/hGh7\x8c\x0brepro_level\x94h\x1d\x90\x8c\x1cget_config_serialization_fns\x94\x8c\x1atorch._dynamo.config_utils\x94hc\x93\x94u.')
torch._inductor.config.load_config(b'\x80\x04\x95\x1c\t\x00\x00\x00\x00\x00\x00}\x94(\x8c\x08__name__\x94\x8c\x16torch._inductor.config\x94\x8c\x07__doc__\x94N\x8c\x0b__package__\x94\x8c\x0ftorch._inductor\x94\x8c\n__loader__\x94\x8c\x1a_frozen_importlib_external\x94\x8c\x10SourceFileLoader\x94\x93\x94)\x81\x94}\x94(\x8c\x04name\x94h\x02\x8c\x04path\x94\x8cd/mnt/petrelfs/xiexinchen/anaconda3/envs/pt2.0/lib/python3.10/site-packages/torch/_inductor/config.py\x94ub\x8c\x08__spec__\x94\x8c\x11_frozen_importlib\x94\x8c\nModuleSpec\x94\x93\x94)\x81\x94}\x94(h\x0ch\x02\x8c\x06loader\x94h\n\x8c\x06origin\x94h\x0e\x8c\x0cloader_state\x94N\x8c\x1asubmodule_search_locations\x94N\x8c\r_set_fileattr\x94\x88\x8c\x07_cached\x94\x8c}/mnt/petrelfs/xiexinchen/anaconda3/envs/pt2.0/lib/python3.10/site-packages/torch/_inductor/__pycache__/config.cpython-310.pyc\x94\x8c\r_initializing\x94\x89ub\x8c\x08__file__\x94h\x0e\x8c\n__cached__\x94h\x1b\x8c\x05debug\x94\x89\x8c\x10disable_progress\x94\x88\x8c\x10verbose_progress\x94\x89\x8c\x0bcpp_wrapper\x94\x89\x8c\x03dce\x94\x89\x8c\x14static_weight_shapes\x94\x88\x8c\x0csize_asserts\x94\x88\x8c\x10pick_loop_orders\x94\x88\x8c\x0finplace_buffers\x94\x88\x8c\x11benchmark_harness\x94\x88\x8c\x0fepilogue_fusion\x94\x89\x8c\x15epilogue_fusion_first\x94\x89\x8c\x0cmax_autotune\x94\x89\x8c\x17realize_reads_threshold\x94K\x04\x8c\x17realize_bytes_threshold\x94M\xd0\x07\x8c\x1brealize_acc_reads_threshold\x94K\x08\x8c\x0ffallback_random\x94\x89\x8c\x12implicit_fallbacks\x94\x88\x8c\rprefuse_nodes\x94\x88\x8c\x0btune_layout\x94\x89\x8c\x11aggressive_fusion\x94\x89\x8c\x0fmax_fusion_size\x94K@\x8c\x1bunroll_reductions_threshold\x94K\x08\x8c\x0ecomment_origin\x94\x89\x8c\tis_fbcode\x94h\x02h7\x93\x94\x8c\x0fcompile_threads\x94K \x8c\x13kernel_name_max_ops\x94K\n\x8c\x0finductor_import\x94\x8c\x0ftorch._inductor\x94\x8c\rshape_padding\x94\x89\x8c\x0epermute_fusion\x94\x89\x8c\x1aprofiler_mark_wrapper_call\x94\x89\x8c\x03cpp\x94}\x94(\x8c\n__module__\x94h\x02\x8c\x07threads\x94J\xff\xff\xff\xff\x8c\x0fdynamic_threads\x94\x89\x8c\x07simdlen\x94N\x8c\x0emin_chunk_size\x94M\x00\x10\x8c\x03cxx\x94N\x8c\x03g++\x94\x86\x94\x8c\x15enable_kernel_profile\x94\x89h\x03Nu\x8c\x06triton\x94}\x94(hBh\x02\x8c\ncudagraphs\x94\x88\x8c\x10debug_sync_graph\x94\x89\x8c\x11debug_sync_kernel\x94\x89\x8c\x0bconvolution\x94\x8c\x04aten\x94\x8c\x0edense_indexing\x94\x89\x8c\tmax_tiles\x94K\x02\x8c\x12autotune_pointwise\x94\x88\x8c tiling_prevents_pointwise_fusion\x94\x88\x8c tiling_prevents_reduction_fusion\x94\x88\x8c\x14ordered_kernel_names\x94\x89\x8c\x18descriptive_kernel_names\x94\x89h\x03Nu\x8c\x05trace\x94}\x94(hBh\x02\x8c\x07enabled\x94\x89\x8c\tdebug_log\x94\x88\x8c\x08info_log\x94\x89\x8c\x08fx_graph\x94\x88\x8c\rir_pre_fusion\x94\x88\x8c\x0eir_post_fusion\x94\x88\x8c\x0boutput_code\x94\x88\x8c\rgraph_diagram\x94\x89\x8c\x0fcompile_profile\x94\x89\x8c\nupload_tar\x94Nh\x03Nu\x8c\x15InductorConfigContext\x94}\x94(hBh\x02\x8c\x0f__annotations__\x94}\x94(\x8c\rstatic_memory\x94\x8c\x08builtins\x94\x8c\x04bool\x94\x93\x94\x8c\x0ematmul_padding\x94hlh+hl\x8c\x12triton_convolution\x94hj\x8c\x03str\x94\x93\x94\x8c\x17rematerialize_threshold\x94hj\x8c\x03int\x94\x93\x94\x8c\x1brematerialize_acc_threshold\x94hsu\x8c\x05_save\x94h\x02\x8c\x1bInductorConfigContext._save\x94\x93\x94\x8c\x06_apply\x94h\x02\x8c\x1cInductorConfigContext._apply\x94\x93\x94\x8c\x08__init__\x94h\x02\x8c\x1eInductorConfigContext.__init__\x94\x93\x94\x8c\t__enter__\x94h\x02\x8c\x1fInductorConfigContext.__enter__\x94\x93\x94\x8c\x08__exit__\x94h\x02\x8c\x1eInductorConfigContext.__exit__\x94\x93\x94h\x03Nu\x8c\x1cget_config_serialization_fns\x94\x8c\x1atorch._dynamo.config_utils\x94h\x84\x93\x94u.')


# REPLACEABLE COMMENT FOR TESTING PURPOSES


# torch version: 2.0.0.dev20230129+cu116
# torch cuda version: 11.6
# torch git version: d6c87398e212e9e632703014776f9e103ef66d60


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Tue_Mar__8_18:18:20_PST_2022 
# Cuda compilation tools, release 11.6, V11.6.124 
# Build cuda_11.6.r11.6/compiler.31057947_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB MIG 3g.40gb : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, primals_1, primals_2):
        amax = torch.ops.aten.amax.default(primals_1, [1], True)
        sub = torch.ops.aten.sub.Tensor(primals_1, amax);  primals_1 = amax = None
        exp = torch.ops.aten.exp.default(sub)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        nll_loss2d_forward = torch.ops.aten.nll_loss2d_forward.default(sub_1, primals_2, None, 0, 255)
        getitem = nll_loss2d_forward[0]
        getitem_1 = nll_loss2d_forward[1];  nll_loss2d_forward = None
        mean = torch.ops.aten.mean.default(getitem);  getitem = None
        mul = torch.ops.aten.mul.Tensor(mean, 1.0);  mean = None
        return [mul, primals_2, sub_1, getitem_1]

args = [((2, 19, 512, 1024), (9961472, 524288, 1024, 1), torch.float32, 'cuda'), ((2, 512, 1024), (524288, 1024, 1), torch.int64, 'cuda')]
args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args]
mod = make_fx(Repro())(*args)


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