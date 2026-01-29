# torch.rand(B, C, H, W, dtype=torch.float32)  # B: batch size, C: channels, H: height, W: width

import torch
import torch.nn as nn
import os

# Set the environment variable to control the ONEDNN primitive cache capacity
os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "1"

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)

    def forward(self, x):
        return self.resnet34(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input shape is (B, C, H, W) where B is the batch size, C is the number of channels,
    # and H, W are the height and width of the image.
    # For ResNet34, the typical input shape is (B, 3, 224, 224).
    # Here we use a variable height to simulate the issue described in the GitHub issue.
    B = 1  # Batch size
    C = 3  # Number of channels (RGB)
    H = torch.randint(50, 800, (1,)).item()  # Random height between 50 and 800
    W = 224  # Fixed width for simplicity
    return torch.rand(B, C, H, W, dtype=torch.float32)

