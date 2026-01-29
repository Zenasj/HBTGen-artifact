# torch.rand(16, 50, 44, 31, dtype=torch.float32), torch.rand(20, 16, 50, 44, 31, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Submodules for both 2D and 3D cases with valid parameters (output_size without zeros)
        self.pool2d = nn.FractionalMaxPool2d(kernel_size=(2,2), output_size=(1,1))
        self.pool3d = nn.FractionalMaxPool3d(kernel_size=(3,2,2), output_size=(1,1,1))
    
    def forward(self, inputs):
        # Process both inputs in parallel
        input2d, input3d = inputs
        return self.pool2d(input2d), self.pool3d(input3d)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate inputs matching the examples in the GitHub issue (with valid parameters)
    input2d = torch.rand(16, 50, 44, 31, dtype=torch.float32)
    input3d = torch.rand(20, 16, 50, 44, 31, dtype=torch.float32)
    return (input2d, input3d)

