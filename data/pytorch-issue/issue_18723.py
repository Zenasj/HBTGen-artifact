# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
from torchvision import models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inception_v3 = models.inception_v3()
        self.inception_v3_pretrained = models.inception_v3(pretrained=True)

    def forward(self, x):
        output_v3 = self.inception_v3(x)
        output_v3_pretrained = self.inception_v3_pretrained(x)
        return output_v3, output_v3_pretrained

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

