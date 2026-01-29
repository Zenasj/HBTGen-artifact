import torch
from torch import nn

# torch.rand(4, 10, 20, device="cuda"), torch.rand(4, 20, 10, device="cuda")  # Inferred input shapes
class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x @ y

def my_model_function():
    return MyModel()

def GetInput():
    batch_size = 4
    seq_length = 10
    input_dim = 20
    x = torch.randn(batch_size, seq_length, input_dim, device="cuda")
    y = torch.randn(batch_size, input_dim, seq_length, device="cuda")
    return (x, y)

