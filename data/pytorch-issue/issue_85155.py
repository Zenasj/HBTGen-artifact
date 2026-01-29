# torch.rand(1, 1, 1, 0, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=1)
    
    def forward(self, x):
        return self.pixel_shuffle(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor with the last dimension as 0 to trigger the reported crash
    return torch.rand(1, 1, 1, 0)

