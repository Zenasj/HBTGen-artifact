# torch.rand(B, 4096, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4096, 4096)  # Large layer to trigger potential memory issues during state_dict collection

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 4096, dtype=torch.float32)

