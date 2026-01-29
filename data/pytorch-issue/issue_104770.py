# torch.randint(2048, (B, S), dtype=torch.long)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2048, 1024)
        self.linear = nn.Linear(1024, 2048)
    
    def forward(self, x):
        return self.linear(self.embedding(x))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(2048, (1, 256), dtype=torch.long)

