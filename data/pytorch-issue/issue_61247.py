# torch.rand(1, 3, 540, 960, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the model components
        self.resize_transform = Resize((540, 960))  # Using torchvision.transforms.Resize for resizing

    def forward(self, x):
        # Apply the resize transformation
        x = self.resize_transform(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 540, 960, dtype=torch.float32)

