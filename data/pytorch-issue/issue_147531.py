# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0]))
        self.batch_size = 128  # Matches original Model's batch size
        
    def forward(self, x):
        # x is a dummy argument required for FSDP compatibility
        return torch.randn(self.batch_size, dtype=self.a.dtype, device=self.a.device) * self.a

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)  # Dummy input to satisfy FSDP's forward signature

