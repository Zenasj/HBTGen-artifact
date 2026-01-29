# torch.rand(1, 3, 100, 100, dtype=torch.float32)  # Inferred input shape from the issue's dummy_in
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(16)
    
    def forward(self, x: torch.Tensor):
        # Add 0 to prevent Conv+BatchNorm fusion (workaround from the issue)
        return self.norm(self.conv(x) + 0.0)  # The +0 ensures ONNX export validity

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = ConvBlock()
        self.conv_block2 = ConvBlock()
    
    def forward(self, x):
        if x.sum() > 0:
            return self.conv_block1(x)
        else:
            return self.conv_block2(x)

def my_model_function():
    # Returns the model with the required structure and workaround
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input shape (1, 3, 100, 100)
    return torch.rand(1, 3, 100, 100, dtype=torch.float32)

