# torch.rand(3, dtype=torch.float32)  # Inferred input shape based on the model's boolean mask size
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bool_tensor", torch.zeros(3,).bool())  # Replicates original issue's buffer
        
    def forward(self, x):
        return x[~self.bool_tensor]  # Reproduces the problematic slicing operation

def my_model_function():
    # Returns the model instance with the same initialization as the issue's TestMod
    return MyModel()

def GetInput():
    # Returns a 1D tensor matching the boolean mask's size (3 elements)
    return torch.rand(3)

