# torch.randint(0, 100, (64, 2048), dtype=torch.long)
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, n_embd = 64, 2048, 4096

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.l1 = nn.Linear(n_embd, 12)
        self.l2 = nn.Linear(12, 11)

    def forward(self, x):
        x0 = self.embed(x)
        x1 = self.ln1(x0)
        x2 = self.l1(x1)
        x3 = F.relu(x2)
        x4 = self.l2(x3)
        return x4

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 100, (B, T), dtype=torch.long)

