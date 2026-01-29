# torch.rand(1, 3, 200, 400, dtype=torch.float32)  # Inferred input shape from the issue's dummy_input
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.aap2d = nn.AdaptiveAvgPool2d(output_size=(output_size, output_size))
        
    def forward(self, inp):
        return self.aap2d(inp)
    
def my_model_function():
    # Returns the model instance with output_size=2 as in the original example
    return MyModel(output_size=2)

def GetInput():
    # Returns a random tensor matching the input shape (B=1, C=3, H=200, W=400)
    return torch.rand(1, 3, 200, 400, dtype=torch.float32)

