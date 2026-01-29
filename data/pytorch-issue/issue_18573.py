# torch.rand(B, 100, dtype=torch.float32)  # Input shape inferred as 2D tensor with features=100
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(100, 50)  # Example layer using the input shape

    def forward(self, x):
        # Example usage of torch.no_grad() context (as shown in the issue's example)
        with torch.no_grad():
            x = self.linear(x)
        return x  # Output from layer executed under no_grad context

def my_model_function():
    # Returns an instance with default initialization
    return MyModel()

def GetInput():
    # Returns a tensor matching the expected input shape with requires_grad (to test context handling)
    return torch.rand(32, 100, dtype=torch.float32, requires_grad=True)

