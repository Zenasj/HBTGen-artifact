# torch.randint(0, 100, (2, 2), dtype=torch.int64)  # Example input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Embedding layer with 101 vocab entries (covers indices 0-100) and embedding dimension 1
        self.embedding_weight = nn.Parameter(torch.randn(101, 1))  # Valid 2D weight tensor
    
    def forward(self, input):
        return F.embedding(input, self.embedding_weight)

def my_model_function():
    # Returns a model instance with valid embedding parameters
    return MyModel()

def GetInput():
    # Generates 2x2 random indices within [0, 100] as per original test case
    return torch.randint(0, 100, (2, 2), dtype=torch.int64)

