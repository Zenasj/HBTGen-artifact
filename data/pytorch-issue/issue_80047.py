# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified EfficientNet-like structure based on common architecture patterns
        self.conv_stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.activation = nn.SiLU()  # EfficientNet typically uses SiLU (swish)
        
        # Example MBConv block (minimal implementation for demonstration)
        self.mbconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),  # Depthwise
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 16, kernel_size=1, bias=False),  # Pointwise
            nn.BatchNorm2d(16),
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 1000)  # Example output size

    def forward(self, x):
        x = self.activation(self.bn1(self.conv_stem(x)))
        x = self.mbconv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def my_model_function():
    # Returns an instance of the simplified EfficientNet model
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected dimensions
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

