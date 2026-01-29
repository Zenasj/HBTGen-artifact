# torch.rand(B, dtype=torch.bfloat16, device='cuda')
import torch
import torch.nn as nn
import torch.distributed as dist

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        y = x * x
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        x = torch.nn.functional.silu(x)
        return x * y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1024, device='cuda', dtype=torch.bfloat16)

