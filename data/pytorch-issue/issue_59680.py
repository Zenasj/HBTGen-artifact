# torch.rand(1, 1, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class ModelA(nn.Module):
    def forward(self, x):
        return torch.neg(torch.relu(torch.relu(x)))

class ModelB(nn.Module):
    def forward(self, x):
        return torch.relu(torch.neg(torch.relu(x)))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = ModelA()
        self.model_b = ModelB()

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        # Return boolean tensor indicating difference
        return torch.tensor(
            not torch.allclose(out_a, out_b, atol=1e-6),
            dtype=torch.bool
        ).view(1)  # Ensure tensor output for compatibility with torch.compile

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 32, 32, dtype=torch.float32)

