# torch.rand(B, dtype=torch.float32, device='cuda')
import torch
import triton
import triton.language as tl

@triton.jit
def triton_(x_ptr, y_ptr, NUMEL: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    data = tl.load(x_ptr + offsets, mask=mask)
    result = data * data

    tl.store(y_ptr + offsets, result, mask=mask)

class MyModel(torch.nn.Module):
    def forward(self, x):
        y = torch.empty_like(x)
        BLOCK_SIZE = 256
        numel = x.numel()
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        triton_[grid](x, y, numel, BLOCK_SIZE)
        return y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(517, dtype=torch.float32, device='cuda')

