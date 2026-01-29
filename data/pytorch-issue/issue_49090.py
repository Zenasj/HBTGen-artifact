# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('switch', torch.tensor([True, False], dtype=torch.bool))

    def forward(self, x):
        # Reproduces the scalar promotion issue with torch.where
        return torch.where(self.switch, x, 0.)

def my_model_function():
    # Returns model instance with fixed boolean condition tensor
    return MyModel()

def GetInput():
    # Returns 1D tensor matching the expected shape/dtype of the model's input
    return torch.rand(2, dtype=torch.float32)

