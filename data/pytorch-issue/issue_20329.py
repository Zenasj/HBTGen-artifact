# torch.randint(0, 8, (B,), dtype=torch.long)
import torch
from torch import nn

n = 8
dim = 10

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(n, dim)
        self.seq = nn.Sequential(
            self.embedding,
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, indices):
        return self.seq(indices)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, n, (3,), dtype=torch.long)

