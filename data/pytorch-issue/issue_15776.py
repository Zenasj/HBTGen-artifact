# torch.rand(1, 1, 352, 720, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_4 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 1, 352, 720, dtype=torch.float32)

