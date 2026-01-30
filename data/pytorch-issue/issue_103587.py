from diffusers import StableDiffusionPipeline
import torch


pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V2.0", torch_dtype=torch.float16
)
pipe.to("cuda:0")
# pipe.unet = torch.compile(pipe.unet, dynamic=False)  # This is OK.
pipe.unet = torch.compile(pipe.unet, dynamic=True)

pipe(prompt="prompt")

from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (s0, s1, s2, s2), (s1*(s2*s2), s2*s2, s2, 1))
    return (Ne(Mod(s2, 8), 0), Ne(Mod(s2, 8), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2
    arg1_1 = 4
    arg2_1 = 64
    arg3_1 = rand_strided((2, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float16)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)

from diffusers import StableDiffusionPipeline
import torch


pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V2.0", torch_dtype=torch.float16
)
pipe.to("cuda:0")
pipe.unet = torch.compile(pipe.unet, dynamic=True)

pipe(prompt="prompt", height=512, width=512)
pipe(prompt="prompt", height=768, width=768)

from diffusers import StableDiffusionPipeline
import torch
import torch._dynamo
import datetime


torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.assume_static_by_default = True


pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V2.0", torch_dtype=torch.float16
)
pipe.to("cuda:0")
pipe.unet = torch.compile(pipe.unet)

start = datetime.datetime.now()
pipe(prompt="prompt", height=512, width=512)
first_done = datetime.datetime.now()
pipe(prompt="prompt", height=768, width=768)
second_done = datetime.datetime.now()

print("first elapsed:", first_done - start)
print("second elapsed:", second_done - first_done)