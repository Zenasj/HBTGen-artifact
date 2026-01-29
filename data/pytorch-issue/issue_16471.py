# torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(3, 3)  # Simple layer to ensure gradients can be computed

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor with requires_grad=True to trigger the grad attribute scenario
    return torch.rand(3, 3, requires_grad=True)

