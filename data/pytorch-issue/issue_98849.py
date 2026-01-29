# torch.rand(5, 10, 16, dtype=torch.float32)  # Inferred input shape from repro
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.d = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.d(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 10, 16)

