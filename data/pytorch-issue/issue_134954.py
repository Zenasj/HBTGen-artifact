# torch.rand(100, 100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, v: float):
        super().__init__()
        self.b = v  # Stored as a float, not a tensor per original code
        
    def forward(self, x):
        return (x.sin() + self.b).sin()

def my_model_function():
    # Returns instance with b=1.0 (as in original test code)
    return MyModel(1.0)

def GetInput():
    # Matches the input shape used in the original test case
    return torch.rand(100, 100, 100, requires_grad=True)

