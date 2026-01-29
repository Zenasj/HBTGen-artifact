# torch.rand(B, C, dtype=torch.float32)
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class MyModel(nn.Module):
    def __init__(self, hidden_dim=10240, bias=False, layers=4):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim, elementwise_affine=bias)
        self.mlps = nn.ModuleList([MLP(hidden_dim, bias) for _ in range(layers)])
        self.post_norm = nn.LayerNorm(hidden_dim, elementwise_affine=bias)

    def forward(self, x):
        x = self.pre_norm(x)
        for mlp in self.mlps:
            x = x + mlp(x)
        x = self.post_norm(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(16, 10240, dtype=torch.float32)

