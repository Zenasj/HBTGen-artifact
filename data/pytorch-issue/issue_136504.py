import torch

import triton
import triton.language as tl

@triton.jit
def triton_(x_ptr, y_ptr, NUMEL: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE*pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    data = tl.load(x_ptr + offsets, mask)
    result = data * data

    tl.store(y_ptr + offsets, result, mask)


def fn(x):
    y = torch.empty_like(x)
    BLOCK_SIZE = 256
    numel = x.numel()
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    triton_[grid](x, y, numel, BLOCK_SIZE)
    return y

x1 = torch.randn(256*2 + 5, device="cuda")
x2 = torch.randn(256*3 + 7, device="cuda")

fn(x1)
fn(x2)

fn_c = torch.compile(fn)

fn_c(x1)
fn_c(x2)