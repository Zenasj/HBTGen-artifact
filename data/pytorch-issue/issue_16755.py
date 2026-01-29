# torch.rand(B, 1) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(1, 2)
        self.l2 = nn.Linear(2, 3)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 1  # Batch size
    return torch.rand(B, 1)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

