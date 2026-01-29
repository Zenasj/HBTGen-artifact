# torch.rand(13, 64, 64, 16, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # LayerNorm configuration matching the problematic case
        self.ln = nn.LayerNorm((16,), eps=1e-6, elementwise_affine=True)
    
    def forward(self, x):
        return self.ln(x)

def my_model_function():
    model = MyModel()
    # Initialize weights to mimic Hiera's small initializer_range=1e-10
    with torch.no_grad():
        model.ln.weight.fill_(1e-10)
        model.ln.bias.fill_(1e-10)
    return model

def GetInput():
    # Input shape and scale matching the issue's problematic scenario
    return torch.rand(13, 64, 64, 16, dtype=torch.float32) * 1e-10

