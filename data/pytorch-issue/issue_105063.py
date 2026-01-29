# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class ConvBNReLU(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=False)
        )

class NestedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.relu1 = nn.ReLU(inplace=False)
        layers = []
        for i in range(3):
            layers.append(ConvBNReLU())
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.features(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nested_module = NestedModule()

    def forward(self, x):
        return self.nested_module(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 224, 224  # Assuming a common input size for image processing
    return torch.rand(B, C, H, W, dtype=torch.float32)

