# torch.rand(10, dtype=torch.complex64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.randn(10, dtype=torch.complex64))  # Reproduces the parameter from the issue
        
    def forward(self, x):
        # x is a dummy input (not used in computation) to satisfy the module interface
        h = torch.ones_like(self.g)
        h[:1] = torch.abs(self.g[:1]).mean()  # Core operation causing dtype casting issues
        return h.mean()  # Loss value to trigger backward pass

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a dummy input tensor (shape/dtype match the model's expected input interface)
    return torch.rand(10, dtype=torch.complex64)

