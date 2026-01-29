import torch
from torch import nn

# torch.rand(1, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.linear(self.linear(x))

def my_model_function():
    model = MyModel()
    with torch.no_grad():
        model.linear.weight[:] = 10.0  # Matches weight initialization in the issue example
    return model

def GetInput():
    return torch.rand(1, dtype=torch.float32)  # Matches input shape (single-element tensor)

