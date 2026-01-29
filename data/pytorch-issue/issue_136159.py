# torch.rand(4, 4, dtype=torch.bfloat16, device='cuda')
import torch
import triton
import triton.language as tl
from triton.runtime.jit import TensorWrapper

torch._inductor.config.cpp_wrapper = True

@triton.jit
def triton_to_fp8(inp_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    vals = tl.load(inp_ptr + offsets, mask)
    fp8_vals = vals.to(tl.float8e4nv)
    tl.store(out_ptr + offsets, fp8_vals, mask)

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = torch.empty(x.shape, device=x.device, dtype=torch.uint8)
        y_fp8 = y.view(dtype=torch.float8_e4m3fn)
        BLOCK_SIZE = 256
        numel = x.numel()
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        triton_to_fp8[grid](x, y_fp8, numel, BLOCK_SIZE)
        # Return boolean indicating correct dtype
        return torch.tensor([y_fp8.dtype == torch.float8_e4m3fn], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 4, dtype=torch.bfloat16, device='cuda')

