# torch.rand(1, 1, 2, 2, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=[5, 5])
        self.conv_transpose.weight.data = torch.ones([1, 2, 5, 5]).contiguous(memory_format=torch.channels_last)

    def forward(self, x):
        return self.conv_transpose(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.ones([1, 1, 2, 2]).contiguous(memory_format=torch.channels_last)

