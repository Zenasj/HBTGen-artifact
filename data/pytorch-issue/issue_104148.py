# torch.rand(8, 5, dtype=torch.float, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct a sequence of 5 Linear layers as per the test's NUM_LINEARS reference
        self.linears = nn.Sequential(
            nn.Linear(5, 5),
            nn.Linear(5, 5),
            nn.Linear(5, 5),
            nn.Linear(5, 5),
            nn.Linear(5, 5),
        )

    def forward(self, x):
        return self.linears(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns input matching the test's (8,5) shape and CUDA device
    return torch.randn(8, 5, dtype=torch.float, device='cuda')

