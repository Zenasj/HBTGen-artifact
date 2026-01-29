# torch.randint(6000, (1024, 80, 20), dtype=torch.int64, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6000, 16)  # Matches original issue's embedding parameters

    def forward(self, x):
        return self.embedding(x)

def my_model_function():
    model = MyModel()
    model.cuda()  # Matches original environment's GPU setup
    return model

def GetInput():
    return torch.randint(6000, (1024, 80, 20), dtype=torch.int64, device='cuda')

