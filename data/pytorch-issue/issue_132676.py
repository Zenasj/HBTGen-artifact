# torch.rand(B, 1, dtype=torch.int8) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.to(torch.float)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 10  # Batch size
    return torch.randint(-128, 127, (B, 1), dtype=torch.int8)

