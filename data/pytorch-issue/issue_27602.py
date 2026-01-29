# torch.rand(1, 3, 416, 416, dtype=torch.float32)  # Input shape for the model

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a simple convolutional layer with "SAME_UPPER" padding
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Apply the convolutional layer
        x = self.conv(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch = 1
    channel = 3
    image_h = 416
    image_w = 416
    img = torch.rand(batch, channel, image_h, image_w, dtype=torch.float32)
    return img

