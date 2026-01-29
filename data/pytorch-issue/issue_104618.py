# torch.rand(1, 2, 5), torch.rand(1, 2, 3, 4), torch.rand(1, 2, 3, 4, 5)  # Input shapes for 1d, 2d, 3d layers
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1d = nn.AdaptiveAvgPool1d(0)
        self.pool2d = nn.AdaptiveAvgPool2d(0)
        self.pool3d = nn.AdaptiveAvgPool3d(0)
    
    def forward(self, inputs):
        x1, x2, x3 = inputs  # Unpack the input tuple
        return (self.pool1d(x1), self.pool2d(x2), self.pool3d(x3))

def my_model_function():
    return MyModel()

def GetInput():
    # Generate inputs for 1d (N,C,L), 2d (N,C,H,W), 3d (N,C,D,H,W)
    input1d = torch.randn(1, 2, 5)
    input2d = torch.randn(1, 2, 3, 4)
    input3d = torch.randn(1, 2, 3, 4, 5)
    return (input1d, input2d, input3d)

