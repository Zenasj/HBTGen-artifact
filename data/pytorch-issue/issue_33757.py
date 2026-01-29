# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn
from torch.optim import AdamW  # This line triggers MyPy error in PyTorch 1.4.0

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.conv(x))

def my_model_function():
    model = MyModel()
    # Example usage of AdamW (demonstrates MyPy error context)
    # optimizer = AdamW(model.parameters(), lr=0.001)
    return model

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

