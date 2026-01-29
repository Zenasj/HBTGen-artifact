# torch.rand(B, 3, 10, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_layer = nn.Sequential(
            nn.Conv2d(3, 3, 1),
            nn.BatchNorm2d(3),
        )
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Dropout(0),
            nn.Flatten(),
            nn.Linear(300, 3),
            nn.BatchNorm1d(3),
        )

    def forward(self, x):
        x = self.res_layer(x)
        x = self.output_layer(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 10, 10)

