import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, v1: torch.Tensor):
        vx = v1.min(dim=1).values # works fine if we remove this operation
        # vx has shape [10, 0]
        v2 = torch.randn_like(vx)
        return v2

model = Model()
x = torch.rand(10, 3, 0) # ERROR
# x = torch.rand(10, 3, 1) # works fine
compiled = torch.compile(model, fullgraph=False)
print(compiled(x))

from ctypes import c_void_p, c_long
import torch
import random
from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()

import triton
import triton.language as tl
from torch._inductor.triton_ops.autotune import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    buf0 = empty_strided((10, 0), (0, 1), device='cpu', dtype=torch.float32)
    del arg0_1
    buf1 = torch.ops.aten.randn_like.default(buf0, dtype=torch.float32, layout=torch.strided, device=device(type='cpu'), pin_memory=False)
    del buf0
    buf2 = buf1
    assert_size_stride(buf2, (10, 0), (1, 1)) # WRONG! should be (10, 0), (0, 1)
    del buf1
    return (buf2, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((10, 3, 0), (3, 1, 1), device='cpu', dtype=torch.float32)
    print_performance(lambda: call([arg0_1]))

import torch

class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, v1: torch.Tensor):
        # v1 has shape [10, 0]
        v2 = torch.randn_like(v1)
        return v2

model = Model()
x = torch.rand(10, 0) # works fine
compiled = torch.compile(model, fullgraph=False)
print(compiled(x))