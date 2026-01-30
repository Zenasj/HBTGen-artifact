from triton.runtime.jit import reinterpret as tl_reinterpret, TensorWrapper  # @manual
import triton.language as tl  # @manual
import torch
import triton
from torch import Tensor
from typing import List, Optional, Tuple, Union
import torch._inductor.config

torch._inductor.config.cpp_wrapper = True

@jit
def triton_to_fp8(inp_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    vals = tl.load(inp_ptr + offsets, mask)
    fp8_vals = vals.to(tl.float8e4nv)
    tl.store(out_ptr + offsets, fp8_vals, mask)

def convert_fn(x):
    y = torch.empty(x.shape, device=x.device, dtype=torch.uint8)
    y_fp8 = y.view(dtype=torch.float8_e4m3fn)
    BLOCK_SIZE = 256
    numel = x.numel()
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    triton_to_fp8[grid](x, y_fp8, numel, BLOCK_SIZE)
    return y_fp8

if __name__ == "__main__":
    x = torch.rand((4, 4), device="cuda", dtype=torch.bfloat16)
    compile_out = torch.compile(convert_fn)(x)
    eager_out = convert_fn(x)
    assert compile_out.dtype == eager_out.dtype, f"{compile_out.dtype} != {eager_out.dtype}"