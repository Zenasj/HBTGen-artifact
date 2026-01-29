# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Inferred input shape based on common image model conventions
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def my_model_function():
    # Returns a simple CNN model compatible with ONNX export
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected dimensions
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

