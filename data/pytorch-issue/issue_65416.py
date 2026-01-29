# torch.rand(1,4,4), torch.rand(1,4,4)  # Two tensors of shape (1,4,4)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.empty(0))  # To track device
    
    def forward(self, inputs):
        A, B = inputs
        device = self.dummy.device
        A = A.to(device)
        B = B.to(device)
        return torch.bmm(A, B)

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.tensor([[
        [0.0000, -143.2371, 0.0000, 0.0000],
        [-143.2371, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, -1.0000, 86.5423],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ]], dtype=torch.float32)
    
    B = torch.tensor([[
        [-0.5000, -0.5000, -0.5000, -0.5000],
        [-0.5000, -0.5000, -0.5000, -0.5000],
        [-0.5000, -0.4980, -0.4961, -0.4941],
        [1.0000, 1.0000, 1.0000, 1.0000]
    ]], dtype=torch.float32)
    
    return (A, B)

