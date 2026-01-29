# torch.rand(67108864, dtype=torch.uint16)  # Input is a uint16 tensor of shape (67108864,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('target', torch.ones(65536 * 1024, dtype=torch.bfloat16, device='cuda'))
        
    def forward(self, source):
        assert source.dtype == torch.uint16, "Source must be uint16"
        assert self.target.dtype == torch.bfloat16, "Target must be bfloat16"
        self.target.view(torch.uint16).copy_(source)
        return self.target

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 256, (65536 * 1024,), dtype=torch.uint16, device='cuda')

