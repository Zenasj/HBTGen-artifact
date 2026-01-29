# torch.rand(1, 16, 64, 72, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.layers(x)

def my_model_function():
    model = MyModel()
    model.eval()  # Matches original issue's eval() setup
    return model

def GetInput():
    return torch.rand(1, 16, 64, 72, dtype=torch.float32)

