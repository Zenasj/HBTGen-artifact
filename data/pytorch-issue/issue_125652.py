# torch.rand(B, 16, 16, dtype=torch.float32)
import torch
from torch import nn

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, dtype=torch.float32))
        self.variance_epsilon = 1e-5  # Scalar constant for numerical stability

    @torch.compile  # Critical to replicate per-layer compilation issue
    def forward(self, hidden_states, residuals=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype), residuals

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(100)])  # 100 layers as in original issue

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states)
        return hidden_states

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(32, 16, 16, dtype=torch.float32)  # Matches original (B, 16, 16) input shape

