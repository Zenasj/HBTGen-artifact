# torch.rand(B, S, dtype=torch.long)  # B=batch_size=1, S=seq_len=16
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        prob = torch.rand([])  # Simulates dynamic control flow causing the error
        if prob < 0.5:
            return x * 2
        else:
            return x * 3

def my_model_function():
    return MyModel()

def GetInput():
    # Matches OPT's input format: token indices tensor
    return torch.randint(0, 10000, (1, 16), device='cuda', dtype=torch.long)

