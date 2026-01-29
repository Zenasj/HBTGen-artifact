# torch.rand(1) ← Input is a dummy tensor; actual size comparison uses fixed n=10
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Submodules are not needed here, but encapsulate comparison logic
        pass

    def forward(self, x):
        n = 10  # Fixed size from the original code example
        empty_tensor = torch.empty(n, dtype=torch.int)
        zeros_tensor = torch.zeros(n, dtype=torch.int)
        # Return a boolean tensor indicating if they are different
        return torch.any(empty_tensor != zeros_tensor).unsqueeze(0).float()

def my_model_function():
    return MyModel()

def GetInput():
    # Dummy input; model's forward does not use it
    return torch.rand(1)

