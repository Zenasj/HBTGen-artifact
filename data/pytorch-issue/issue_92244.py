# torch.rand(8, 1, 4, 4, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.layer1 = nn.Conv2d(1, 256, kernel_size=2, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv2d(256, num_classes, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = self.layer2(out)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(8, 1, 4, 4, dtype=torch.float32)

