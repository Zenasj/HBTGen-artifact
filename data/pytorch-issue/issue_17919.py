# torch.randint(0, 10, (3,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(10, 3, sparse=True)  # Matches issue's embedding parameters
        
    def forward(self, x):
        return self.embedding(x).sum()  # Reproduces the sum() operation from the issue examples

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    return torch.randint(0, 10, (3,), dtype=torch.long)  # Generates valid indices for embedding (size 10)

