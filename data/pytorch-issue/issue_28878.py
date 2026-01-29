# torch.rand(1, 1, dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 64)
        self.fc1 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.emb(x)
        return self.fc1(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (1, 1), dtype=torch.long)

