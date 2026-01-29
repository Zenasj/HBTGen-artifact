# torch.rand(B, C, H, W, D, dtype=torch.float32)  # Input shape: (batch_size, channels, height, width, depth)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm3d(3, eps=0.001, momentum=0.7, affine=False)
    
    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Shape: (batch_size, channels, height, width, depth)
    return torch.randn(2, 3, 4, 4, 4, device='cuda', requires_grad=True).contiguous(memory_format=torch.channels_last_3d)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# grad_out = torch.randn(output.shape, device='cuda').contiguous(memory_format=torch.channels_last_3d)
# output.backward(grad_out)
# for gi in input_tensor.grad:
#     print("gi.shape: ", gi.shape, " gi.stride: ", gi.stride())

