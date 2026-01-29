# torch.rand(1)  # Dummy input tensor (shape not used in computation)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.rand(10, 20, requires_grad=True))  # Tensor requiring gradients
        self.register_buffer('y', torch.rand(10))  # Non-parameter tensor for in-place op

    def forward(self, _input):
        # Division operation using self.y (non-grad tensor)
        z = self.x / self.y.unsqueeze(1)
        # In-place operation on self.y after its use in z - causes gradient computation error
        self.y.abs_()
        return z.sum()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input (unused in computation but required for interface)

