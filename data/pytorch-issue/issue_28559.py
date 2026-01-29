# torch.rand(B, D, dtype=torch.float32)  # Assuming input shape (batch, features)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        encoder_norm = nn.ReLU()
        self.encoder = N(encoder_norm)

    def forward(self, x):
        return self.encoder(x)

class N(nn.Module):
    __constants__ = ['norm']

    def __init__(self, norm=None):
        super(N, self).__init__()
        self.activation = torch.nn.functional.relu  # This line causes the JIT error
        self.norm = norm

    def forward(self, src):
        output = src
        if self.norm is not None:
            output = self.norm(output)
        return output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10, dtype=torch.float32)

