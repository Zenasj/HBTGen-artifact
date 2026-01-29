# torch.rand(B, 3, 28, 28, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN structure to demonstrate FSDP usage
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 28 * 28, 10)  # For 28x28 input images

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # 4x3x28x28 input matching the CNN's expected dimensions
    return torch.rand(4, 3, 28, 28, dtype=torch.float32)

