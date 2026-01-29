# torch.rand(3, dtype=torch.float32, device='cuda')  # Assumed input is a 1D tensor of shape (3,)
import torch
import triton
import triton.language as tl
from torch import nn
from torch.autograd import Function

class MySin(Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x)
        add_one(x, out)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        saved, = ctx.saved_tensors
        out = torch.empty_like(grad)
        add_one(saved, out)
        return out

@triton.jit
def add_one_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = x + 1
    tl.store(out_ptr + offsets, output, mask=mask)

def add_one(x, out):
    n_elements = x.numel()
    add_one_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

class MyModel(nn.Module):
    def forward(self, x):
        return MySin.apply(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32, device='cuda')

