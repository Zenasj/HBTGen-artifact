# torch.rand(2, 12, 16, 32, 32, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize indices with fixed values for reproducibility
        i1 = torch.arange(2).unsqueeze(-1)  # (2,1)
        # Generate i2 with deterministic seed
        torch.manual_seed(0)
        rand_tensor = torch.rand(2, 12)
        indices = torch.argsort(rand_tensor, dim=-1)
        i2 = indices[:, :3]  # (2,3)
        self.register_buffer('i1', i1)
        self.register_buffer('i2', i2)

    def forward(self, x):
        return x[self.i1, self.i2]

def my_model_function():
    return MyModel()

def GetInput():
    # Create input tensor with shape (2,12,16,32,32) and channels_last memory format
    x = torch.randn((24, 16, 32, 32), dtype=torch.float32).to(memory_format=torch.channels_last)
    x = x.view(2, 12, 16, 32, 32)
    return x

