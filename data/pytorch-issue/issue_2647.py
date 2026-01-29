# torch.rand(B, 2, dtype=torch.float32)  # Inferred input shape for a Linear(2,1) model
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(2, 1)  # Matches the original Linear(2,1) model in the issue

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    B = 1  # Batch size (matches the issue's example of 1x2 input)
    return torch.rand(B, 2, dtype=torch.float32)  # Matches the issue's torch.randn(1,2)

