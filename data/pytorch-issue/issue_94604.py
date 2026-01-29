# torch.rand(1, 3, 960, 960, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.InstanceNorm2d(num_features=3)  # Matches input channels=3

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    model = MyModel()
    model.eval()  # Matches original issue's evaluation mode
    return model

def GetInput():
    return torch.randn(1, 3, 960, 960, dtype=torch.float32)

