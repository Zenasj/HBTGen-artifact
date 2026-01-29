# torch.rand(32, 3, 64, 64, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(64 * 64 * 64, 100)  # Flattened size after Conv2d
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.relu(self.linear(x))

def my_model_function():
    # Return initialized model on CUDA device
    return MyModel().to("cuda:0")

def GetInput():
    # Return input tensor matching the model's expected dimensions and device
    return torch.randn(32, 3, 64, 64, dtype=torch.float32).to("cuda:0")

