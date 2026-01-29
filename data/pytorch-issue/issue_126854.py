# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(10, 1)  # Matches input dimension from the issue's example

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    # Returns the model instance used in the issue's example
    return MyModel()

def GetInput():
    # Generates input matching the model's expected shape (B, 10)
    return torch.rand(1, 10, dtype=torch.float32)  # B=1 as minimal test case

