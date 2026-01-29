import torch
import torch.nn as nn

# torch.rand(B, 3, 224, 224, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 224 * 224, 10)  # Matches input shape after flattening

    def forward(self, x):
        with torch.profiler.record_function("conv_block"):
            x = self.conv(x)
            x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flattening for linear layer
        with torch.profiler.record_function("fc_block"):
            x = self.fc(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

