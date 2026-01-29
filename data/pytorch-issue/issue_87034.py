# torch.rand(3, 4, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.weights = torch.nn.Parameter(
            torch.nested.nested_tensor(
                [torch.empty(size, dtype=torch.float) for size in sizes]
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for bit in self.weights.unbind():
                torch.nn.init.normal_(bit)

    def forward(self, x):
        # Forward pass that returns input (focus is on parameter handling)
        return x

def my_model_function():
    # Returns MyModel with default sizes [(3,4), (2,5)]
    return MyModel(sizes=[(3, 4), (2, 5)])

def GetInput():
    # Returns a compatible input tensor (shape (3,4))
    return torch.rand(3, 4)

