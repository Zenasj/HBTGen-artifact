# torch.rand(B, C, H, W, dtype=torch.float16)  # Inferred input shape: (2, 8, 4, 4) with dtype=torch.float16

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(8, 4, 3).cuda().half()
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    input = torch.randint(1, 10, (2, 8, 4, 4), dtype=torch.float32, requires_grad=True)
    input = input.to(device="cuda", memory_format=torch.channels_last, dtype=torch.float16)
    return input

