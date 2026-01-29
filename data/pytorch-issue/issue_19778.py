# torch.rand(2, dtype=torch.float32)  # Input shape is (2,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Non-view case: initialized with correct shape (2,1,1)
        self.x_non_view = nn.Parameter(torch.tensor([[[0.25]], [[0.75]]], requires_grad=True))
    
    def forward(self, input):
        # Create the view from the input (shape (2,) -> (2,1,1))
        x_view = input.view(2, 1, 1)
        return self.x_non_view, x_view  # Return both tensors for gradient comparison

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor of shape (2,)
    return torch.rand(2, requires_grad=True)

