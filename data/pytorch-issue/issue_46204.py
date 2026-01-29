# torch.rand(2, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # This in-place modification of a view of a leaf tensor with requires_grad=True
        # will now raise an error immediately (BC-breaking change)
        view = x[0]  # Create a view of the first element
        view.add_(1)  # In-place operation on the view
        return x

def my_model_function():
    # Returns the model that demonstrates the BC-breaking in-place view modification
    return MyModel()

def GetInput():
    # Returns a 1D tensor of shape (2,) with requires_grad=True to trigger the error
    return torch.rand(2, requires_grad=True)

