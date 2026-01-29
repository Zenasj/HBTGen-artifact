import torch
from torch import nn

# torch.rand(B, 1, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 10, bias=True)  # Avoids None bias
        self.x = torch.tensor(3.0, dtype=torch.float32)  # Use tensor instead of scalar
        self.index = torch.tensor(0, dtype=torch.long)  # Tensor index for slicing

    def forward(self, x):
        # Comparison part (x < self.x)
        comp = x < self.x
        # Linear layer with explicit bias
        lin_out = self.linear(x)
        # Slicing using tensor-based index
        sliced = lin_out[self.index]
        return comp, lin_out, sliced

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, dtype=torch.float32)

