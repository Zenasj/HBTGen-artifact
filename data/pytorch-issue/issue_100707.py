# torch.rand(N, dim, dtype=torch.cfloat) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(dim, dtype=torch.cfloat))
    
    def forward(self, x):
        return torch.abs(x @ self.weight)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(5)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    N = 100
    dim = 5
    x = torch.rand((N, dim), dtype=torch.cfloat).to("cuda")
    return x

