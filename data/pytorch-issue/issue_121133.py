# torch.rand(3, dtype=torch.float32)  # Inferred input shape from original example
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh_layer = nn.Tanh()  # Autograd-based tanh

    def forward(self, x):
        self.x = x  # Store input to access gradients later
        tanh_out = self.tanh_layer(x)
        self.manual_derivative = 1 - tanh_out ** 2  # Manual derivative (1 - tanh²)
        return tanh_out

    def compare_gradients(self):
        """Compares autograd gradient with manual derivative using torch.allclose"""
        if self.x.grad is None:
            raise RuntimeError("Gradient not computed (run backward first)")
        return torch.allclose(self.manual_derivative, self.x.grad)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32, requires_grad=True)

