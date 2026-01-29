# torch.randint(0, 500000, (300, L), dtype=torch.long)
import torch
import random
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(500000, 100)
        self.conv = nn.Conv1d(100, 200, 3)

    def forward(self, x):
        x = self.embed(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        return torch.mean(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 300  # Batch size fixed at 300 as per original example
    L = random.randint(100, 3000)  # Variable sequence length
    return torch.randint(0, 500000, (B, L), dtype=torch.long)

