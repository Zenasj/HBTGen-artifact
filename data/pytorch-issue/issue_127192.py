# torch.rand(B, 300, 5, dtype=torch.float32)  # Inferred from example input shape (1, 300, 5)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dim):
        super(MyModel, self).__init__()
        self.input_dim = dim
        self.l1 = nn.Linear(dim, 300, bias=False)
        self.b1 = nn.BatchNorm1d(300)

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten all dimensions except last
        h = self.l1(x)
        h = self.b1(h)
        return h

def my_model_function():
    # Initialize with dim=5 as in the original example
    return MyModel(5)

def GetInput():
    # Matches the input shape used in the original example (1, 300, 5)
    return torch.randn(1, 300, 5)

