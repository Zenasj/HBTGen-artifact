# torch.rand(2, dtype=torch.float32)  # Input shape is (2,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.ones(3, 2))  # 3x2 matrix from original code

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(-1)  # Convert (2,) → (2, 1)
        matmul_result = self.weight.matmul(x_unsqueezed)  # (3,2) @ (2,1) → (3,1)
        squeezed = matmul_result.squeeze(-1)  # (3,)
        summed = squeezed.sum()  # Scalar result
        return summed

def my_model_function():
    return MyModel()

def GetInput():
    return torch.ones(2, requires_grad=True)  # Matches original input shape and grad requirement

