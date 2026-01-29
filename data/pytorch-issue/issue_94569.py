# torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Example input shape for a typical CNN-like model
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.linear = nn.Linear(16*224*224, 10)  # Example head for classification

    def forward(self, x, optional=None):
        # Example usage of optional parameter that may be None during tracing
        if optional is not None:
            x = x + optional  # Simple operation to demonstrate argument handling
        x = self.conv(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

