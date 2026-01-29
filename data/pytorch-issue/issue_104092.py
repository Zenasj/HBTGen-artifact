# torch.randint(5, (), dtype=torch.int64)
import torch
from torch import nn

class HasCustomIndexing:
    def __init__(self):
        self.l = list(range(5))
    
    def __getitem__(self, item):
        return self.l[item]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.indexer = HasCustomIndexing()
    
    def forward(self, index_tensor):
        # The __getitem__ here triggers Dynamo's issue with tensor-based indexing
        value = self.indexer[index_tensor]
        return torch.tensor([value], dtype=torch.long)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(5, (), dtype=torch.int64)

