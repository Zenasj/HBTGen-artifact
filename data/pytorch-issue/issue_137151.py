# torch.randint(0, 32_000, (B, 2048), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32_000, 1024)
        self.linear = nn.Sequential(*[nn.Linear(1024, 1024) for _ in range(50)])
        self.lm_head = nn.Linear(1024, 32_000)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.linear:
            x = layer(x)
        x = self.lm_head(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 32_000, (1, 2048), dtype=torch.long).cuda()

