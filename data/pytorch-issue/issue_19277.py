# torch.randint(4 * 10**6, (B,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        dim = 5
        n = 4 * 10 ** 6
        self.emb = nn.Embedding(n, dim)
        self.lin1 = nn.Linear(dim, 1)
        self.seq = nn.Sequential(
            self.emb,
            self.lin1,
        )

    def forward(self, input):
        return self.seq(input)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size (adjustable)
    return torch.randint(4 * 10**6, (B,), dtype=torch.long)

