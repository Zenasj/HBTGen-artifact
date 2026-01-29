# torch.rand(4, 480, 1024, dtype=torch.float32)  # Shape from the non-power-of-2 input case
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.Linear(1024, 1024, bias=False)
        )
    
    def forward(self, x):
        return self.layers(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel().cuda()  # Explicitly move to CUDA as in the issue's setup

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.rand(4, 480, 1024, dtype=torch.float32).cuda()

