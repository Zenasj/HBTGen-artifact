# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.celu_layer = nn.CELU(alpha=2)  # Matches the model in the issue's reproduction code
        
    def forward(self, x):
        return self.celu_layer(x)

def my_model_function():
    return MyModel()  # Directly instantiate the model with default parameters

def GetInput():
    return torch.rand(1)  # Matches the input shape used in the issue's reproduction

