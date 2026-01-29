# torch.rand(1, dtype=torch.float32)  # Input is a scalar to trigger forward computation
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a large shared memory tensor that may cause /dev/shm OOM
        self.large_tensor = torch.Tensor(1000000000).share_memory_()
    
    def forward(self, x):
        # Forward pass simply multiplies input by first element of large_tensor
        return x * self.large_tensor[0]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)

