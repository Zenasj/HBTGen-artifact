# torch.rand(100, 100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Simulate operations requiring thread initialization
        return torch.bmm(x, x).sum()

def my_model_function():
    # Explicitly set num_threads to trigger CPU info initialization
    torch.set_num_threads(8)
    return MyModel()

def GetInput():
    # Input tensor matching the model's expected dimensions
    return torch.rand(100, 100, 100, dtype=torch.float32)

