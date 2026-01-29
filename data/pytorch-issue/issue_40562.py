# torch.rand(B, 128, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(128, 128)

    def forward(self, input_ids):
        output = self.model(input_ids)
        self.model(input_ids)  # Second call
        output = self.model(input_ids)  # Third call (overwrites output)
        self.model(input_ids)  # Fourth call (no output)
        return output  # Returns result of third call

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size placeholder
    return torch.rand(B, 128, dtype=torch.float32)

