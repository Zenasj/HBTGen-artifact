# torch.rand(2, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)  # Matches the minimal example's tensor shape

    def forward(self, x):
        x = self.fc(x)
        indices = torch.argmin(x, dim=1)  # Triggers the argmin indexing issue
        selected = x[indices]  # Problematic indexing operation causing backward error
        return selected.sum()  # Returns a scalar for loss computation

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2)  # 2x2 input matching the model's forward requirements

