# torch.rand(1,) ← Dummy input tensor as the model doesn't use it
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tmp = None  # Stores the tensor with decreasing size

    def forward(self, x):
        # Simulate the loop's tensor creation with memory retention
        if self.tmp is None:
            self.tmp = torch.zeros((5000, 1000000), dtype=torch.float32, device='cpu')
        else:
            # Reduce columns by 10,000 each call, reusing prior memory (retaining initial allocation)
            new_cols = self.tmp.size(1) - 10000
            self.tmp = torch.zeros((5000, new_cols), out=self.tmp, device='cpu')
        return x  # Model returns input unused, focusing on tensor creation side-effect

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input to satisfy model's interface requirements
    return torch.rand(1, dtype=torch.float32)

