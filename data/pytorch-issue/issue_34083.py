# torch.rand(B, C, H, W, dtype=torch.float16, device='cuda').contiguous(memory_format=torch.channels_last)  # (200, 512, 28, 28)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False)

    def forward(self, x):
        return self.max_pool(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(200, 512, 28, 28, device='cuda', dtype=torch.float16).contiguous(memory_format=torch.channels_last)

