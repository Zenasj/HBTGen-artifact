# torch.randint(0, 1, (B,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(1, 1, sparse=True)
    
    def forward(self, x):
        return self.embedding(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Example batch size
    return torch.randint(0, 1, (B,), dtype=torch.long)

