# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)  # Example layer to define input shape
        # Duplicate parameter to trigger the ASGD warning
        self.duplicate_param = nn.Parameter(torch.randn(5))
        self.duplicate_param2 = self.duplicate_param  # Same parameter registered twice

    def forward(self, x):
        x = self.fc(x)
        # Use parameters to ensure they are part of the computation graph
        return x + self.duplicate_param.sum() + self.duplicate_param2.sum()

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Example batch size
    return torch.rand(B, 10, dtype=torch.float32)

