# torch.rand(1, 5, 10, 10, dtype=torch.float32)  # Inferred input shape (B=1, C=5, H=10, W=10)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        device = x.device
        output = torch.multinomial(x.view(-1), 5)  # Samples 5 indices from flattened input
        return output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 5, 10, 10)  # Random tensor matching input shape requirements

