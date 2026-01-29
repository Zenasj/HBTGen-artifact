# torch.rand(B, 10, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, inshape=10, outshape=1):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inshape, outshape)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

def my_model_function():
    # Returns an MLP with input size 10 and output size 1
    return MyModel()

def GetInput():
    # Returns a random tensor of shape (1, 10)
    return torch.rand(1, 10, dtype=torch.float)

