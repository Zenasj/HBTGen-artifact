# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)  # Minimal trainable layer to reproduce parameter generator issue

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns 2D tensor matching Linear layer input (batch_size, in_features)
    return torch.rand(2, 10, dtype=torch.float32)

