# torch.rand(2), torch.rand(2)  # input shape for MyModel is a tuple of two tensors of shape (2,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.grads = []  # Stores gradients from hook
    
    def hook(self, grad):
        # Reproduces the problematic list append operation
        self.grads.append(grad)
        return grad
    
    def forward(self, inputs):
        x, y = inputs
        # Register hook on x's gradient during forward
        x.register_hook(lambda grad: self.hook(grad))
        return x + y  # Forward pass matches original fn(x, y) = x + y

def my_model_function():
    return MyModel()

def GetInput():
    # Returns two tensors matching the input requirements of MyModel
    x = torch.rand(2, requires_grad=True)
    y = torch.rand(2, requires_grad=True)
    return (x, y)

