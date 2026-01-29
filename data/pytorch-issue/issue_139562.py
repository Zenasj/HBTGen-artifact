# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = nn.Identity()  # Placeholder for first model
        self.model_b = nn.Identity()  # Placeholder for second model

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        # Compare outputs using torch.isclose with rtol and atol as integers (valid case from the issue)
        return torch.isclose(out_a, out_b, rtol=3, atol=4)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

