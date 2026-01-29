# torch.rand(1, 2, dtype=torch.float32)  # Inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.rand(2, 2))

    def forward(self, x):
        y1 = x @ self.weight
        y2 = torch.einsum('...x,xy->...y', [x, self.weight])
        return y1, y2

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    _x = torch.rand(1, 2, device=device)
    x = _x.expand(2, -1).requires_grad_()
    return x

