# torch.rand(1, 16, 12, 12, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)

    def forward(self, x):
        h = self.downsample(x)
        output = self.upsample(h, output_size=x.size())
        return output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 16, 12, 12, dtype=torch.float32)

