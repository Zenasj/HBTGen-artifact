# torch.rand(10, 10, dtype=torch.float32).cuda()
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 10)  # Matches the Linear layer in the issue's example
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns the model with default initialization
    return MyModel()

def GetInput():
    # Returns a CUDA tensor matching the input shape expected by the model
    return torch.rand(10, 10, dtype=torch.float32).cuda()

