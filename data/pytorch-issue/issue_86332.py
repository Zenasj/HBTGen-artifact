# torch.rand(1, 3456, 512, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters fixed based on the repro script's func() arguments
        self.o_size = [512, 1]
        self.k_size = [9, 1]
        self.dilation = [1, 1]
        self.padding = [4, 0]
        self.stride = [1, 1]

    def forward(self, x):
        return torch.ops.aten.col2im(
            x,
            self.o_size,
            self.k_size,
            self.dilation,
            self.padding,
            self.stride
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3456, 512, dtype=torch.float32)

