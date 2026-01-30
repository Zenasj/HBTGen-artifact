import numpy as np

import torch

def calculate_scale(inp):
    amax = torch.abs(torch.max(inp))
    scale = 448.0 / torch.clamp(amax, min=1e-12)
    scale = scale.to(torch.float32)
    return scale

dtype = torch.bfloat16
torch.manual_seed(0)
inp = torch.randn(16, 16, 768, dtype=dtype, device="cuda")
eager_scale = calculate_scale(inp)
compile_scale = torch.compile(calculate_scale)(inp)
torch.testing.assert_close(eager_scale, compile_scale)

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (16, 16, 768), (12288, 768, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((24, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_red_fused_max_0.run(arg0_1, buf0, 24, 8192, grid=grid(24), stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [max_1, amax, clamp, res, scale], Original ATen: [aten.max, aten.abs, aten.clamp, aten.reciprocal, aten.mul, aten._to_copy]
        triton_per_fused__to_copy_abs_clamp_max_mul_reciprocal_1.run(buf0, buf2, 1, 24, grid=grid(1), stream=stream0)
        del buf0
    return (buf2, )

import torch

def calculate_scale(inp):
    amax = torch.abs(torch.max(inp))
    scale = 448.0 / torch.clamp(amax, min=1e-12)
    scale = scale.to(torch.float32)
    return scale

dtype = torch.bfloat16
torch.manual_seed(0)
inp = torch.randn(16, 16, 768, dtype=dtype, device="cuda")
eager_scale = calculate_scale(inp)
compile_scale = torch.compile(calculate_scale)(inp)
fp32_scale = calculate_scale(inp.to(torch.float))

torch.testing.assert_close(fp32_scale, compile_scale)

print("Max divergence from fp32, compile: ", torch.max(torch.abs(fp32_scale - compile_scale)))
# 0.
print("Max divergence from fp32, eager: ", torch.max(torch.abs(fp32_scale - eager_scale)))
# 0.1423

torch._dynamo.reset()

from torch._inductor import config
config.emulate_precision_casts = True

compile_scale = torch.compile(calculate_scale)(inp)
torch.testing.assert_close(eager_scale, compile_scale)
# passes

import torch

def calculate_scale(inp):
    amax = torch.abs(torch.max(inp))
    scale = 448.0 / torch.clamp(amax, min=1e-12)
    scale = scale.to(torch.float32)
    return scale

dtype = torch.bfloat16
torch.manual_seed(0)
inp = torch.randn(16, 16, 768, dtype=dtype, device="cuda")
eager_scale = calculate_scale(inp)
compile_scale = torch.compile(calculate_scale)(inp)
fp32_scale = calculate_scale(inp.to(torch.float))

torch.testing.assert_close(fp32_scale, compile_scale)

print("Max divergence from fp32, compile: ", torch.max(torch.abs(fp32_scale - compile_scale)))
# 0.
print("Max divergence from fp32, eager: ", torch.max(torch.abs(fp32_scale - eager_scale)))
# 0.1423

torch._dynamo.reset()

from torch._inductor import config
config.emulate_precision_casts = True

compile_scale = torch.compile(calculate_scale)(inp)
torch.testing.assert_close(fp32_scale, compile_scale)
# passes