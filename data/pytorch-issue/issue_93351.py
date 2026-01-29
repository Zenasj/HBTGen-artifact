# torch.rand(1, 2, 2, 1, dtype=torch.float64) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn.functional as F

class MyModel(torch.nn.Module):
    def forward(self, x):
        v1 = F.pad(x, pad=(1, 0))
        return torch.gt(v1, x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 2, 2, 1, dtype=torch.float64)

