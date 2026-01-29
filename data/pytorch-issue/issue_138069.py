# torch.rand(2, dtype=torch.float32)  # Input shape is (2,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Integer tensors to trigger int_truediv during symbolic execution
        self.i = torch.tensor(2, dtype=torch.int32)
        self.divisor = torch.tensor(2, dtype=torch.int32)

    def forward(self, x):
        # Perform division between two integer tensors to trigger the error
        intermediate = self.i / self.divisor  # This uses truediv between int tensors
        val = intermediate * 123.4567
        rounded_val = round(val.item(), 1)  # Emulate original test case's round()
        return x + rounded_val

def my_model_function():
    # Returns the model instance with predefined parameters
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input shape (2,)
    return torch.rand(2, dtype=torch.float32)

