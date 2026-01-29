# torch.randint(0, 5, (1, 2), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(5, 1, padding_idx=0, sparse=True)
    
    def forward(self, x):
        return self.embedding(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor of zeros (padding indices) to trigger the SparseAdam bug
    return torch.zeros((1, 2), dtype=torch.long)

