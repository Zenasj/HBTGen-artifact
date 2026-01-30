ms = ms - benchmarker.benchmark_gpu(lambda: wrapped_jit_function.clone_args(*args))

import logging
import torch
from transformers import BertModel, BertTokenizer

def compile_and_run_bert(device="cpu"):
    model_name = "bert-base-uncased"
    # Copy pasted from here https://huggingface.co/bert-base-uncased
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model = torch.compile(model)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors="pt").to(device)

    # Run compiled model (make sure all set)
    return model(**encoded_input)


if __name__ == "__main__":
    # For debugging
    torch._logging.set_logs(
        fusion=True, inductor=logging.DEBUG
    )
    torch._inductor.config.debug = True
    
    # Model compilation and benchmarking
    torch._inductor.config.benchmark_fusion = True
    output = compile_and_run_bert(device="cuda:6")
    print(output)

import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid

@triton_heuristics.persistent_reduction(
    size_hints=[16, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=6, cc=70, major=7, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=80), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'Placeholder.DESCRIPTIVE_NAME', 'mutated_arg_names': [], 'no_x_dim': True, 'num_load': 3, 'num_reduction': 3, 'backend_hash': '7A009071100551DC6842B77D79D5764EC36BF3F0A7F051A8A7AE82D3D9F62463', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'kernel_num_gb': 0.0}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel):
    xnumel = 12
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tl.store(out_ptr0 + (x0), tmp20, xmask)


def get_args():
    arg_0 = rand_strided((12, 768), (768, 1), device='cuda:6', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:6', dtype=torch.float32)
    arg_2 = rand_strided((1, 12, 768), (9216, 768, 1), device='cuda:6', dtype=torch.float32)
    arg_3 = rand_strided((1, 12, 1), (12, 1, 12), device='cuda:6', dtype=torch.float32)
    return arg_0, arg_1, arg_2, arg_3,


def call(args):
    with torch.cuda._DeviceGuard(6):
        torch.cuda.set_device(6)
        stream6 = get_raw_stream(6)
        triton_.run(*args, 12, 768, grid=grid(12), stream=stream6)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(6):
        torch.cuda.set_device(6)
        return triton_.benchmark_all_configs(*args, 12, 768, grid=grid(12))


if __name__ == '__main__':
    from triton.testing import do_bench

    args = get_args()
    ms = do_bench(lambda: call(args), rep=40, fast_flush=True)
    num_gb = 0.0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")