# torch.rand(1, 2, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def forward(self, x):
        return F.dropout(x, p=0.5)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.original = Model()
        self.original.eval()  # Ensure eval mode as in the original issue
        self.compiled = torch.compile(self.original)  # Encapsulate both models

    def forward(self, x):
        original_out = self.original(x)
        compiled_out = self.compiled(x)
        # Return a boolean tensor indicating if outputs differ
        difference = not torch.allclose(original_out, compiled_out)
        return torch.tensor(difference, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 2)

