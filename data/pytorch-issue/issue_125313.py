# torch.rand(1, 10, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.scale = nn.Parameter(torch.randn(1, 10))  # Initialized as parameter for model tracking

    def forward(self, x):
        # Reshape 4D input to 2D for linear layer compatibility
        x = x.view(x.size(0), -1)  
        x = self.linear1(x)
        torch._dynamo.graph_break()  # Preserve graph break from original issue's code
        return F.relu(x) * self.scale

def my_model_function():
    # Return model instance with initialized parameters
    return MyModel()

def GetInput():
    # Generate 4D input matching the model's expected dimensions
    return torch.rand(1, 10, 1, 1, dtype=torch.float32)

