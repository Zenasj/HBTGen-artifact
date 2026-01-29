# torch.rand(B, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example parameter using nn.Parameter with default requires_grad=True
        self.weight = nn.Parameter(torch.randn(10, 5))
    
    def forward(self, x):
        return torch.mm(x, self.weight.t())

def my_model_function():
    return MyModel()

def GetInput():
    # Random input matching the model's expected input shape (batch_size, 5)
    return torch.rand(2, 5, dtype=torch.float32)

