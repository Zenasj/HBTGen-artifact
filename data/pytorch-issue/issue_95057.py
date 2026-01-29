# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 5)  # Example layer before logsigmoid

    def forward(self, x):
        x = self.fc(x)
        # Replace F.logsigmoid with alternative implementation to avoid forward AD error
        return -F.softplus(-x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3)  # Batch=2, input features=3

