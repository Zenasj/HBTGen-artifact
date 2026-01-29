# torch.rand(1, 3, 20, 20, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, hidden_states):
        # Uses nearest interpolation causing decomposition issues
        return F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape from the issue's example
    return torch.rand(1, 3, 20, 20, dtype=torch.float32)

