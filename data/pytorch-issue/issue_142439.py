# torch.rand(2, 1, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(1, 1) for _ in range(2)])

    def forward(self, x):
        for fc in self.fcs[:1]:
            x = fc(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, 1, dtype=torch.float32)

