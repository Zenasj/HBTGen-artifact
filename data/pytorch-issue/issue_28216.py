# torch.rand(B, 2, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(2, 1)  # Matches test case's Linear(2,1) model

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel()  # Returns initialized model instance

def GetInput():
    # Returns random input tensor matching Linear(2,1) input requirements
    return torch.rand(32, 2, dtype=torch.float)  # Batch size 32, 2 features

