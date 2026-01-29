# torch.rand(32, 50, 768, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=False)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, inp):
        matmul_output = inp @ self.weight
        return self.layer_norm(matmul_output)

def my_model_function():
    return MyModel(hidden_size=768)

def GetInput():
    batch_size = 32
    seq_length = 50
    hidden_size = 768
    return torch.randn(batch_size, seq_length, hidden_size)

