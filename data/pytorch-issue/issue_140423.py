# torch.rand(1024, dtype=torch.float16, device='cuda') * 0.01
import torch
import triton
import triton.language as tl
from torch import nn

def get_config():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=1))
    configs.append(triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=1))
    configs.append(triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=2))
    configs.append(triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=2))
    return configs

@triton.autotune(configs=get_config(), key=['n_elements'])
@triton.jit
def kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel[grid](x, y, output, n_elements)
    return output

class CustomLayer(nn.Module):
    def forward(self, x):
        return op(x, x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            CustomLayer(),
            CustomLayer(),
            CustomLayer(),
            CustomLayer(),
        )
        
    def forward(self, x):
        return self.layers(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1024, dtype=torch.float16, device='cuda') * 0.01

