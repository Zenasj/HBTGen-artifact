# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64))
        self.bias = nn.Parameter(torch.randn(64))
        self.register_buffer('running_mean', torch.randn(64))
        self.register_buffer('running_var', torch.randn(64))

    def forward(self, x):
        out, _, _ = torch.native_batch_norm(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            training=True,
            momentum=0.1,
            eps=1e-5
        )
        return torch.relu(out)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(32, 64, 122, 122, dtype=torch.float32)

