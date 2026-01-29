# torch.randint(0, 10, (B, 4), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)  # Matches test's embedding_matrix shape (10,3)

    def forward(self, x):
        return torch.nn.functional.embedding(x, self.embedding.weight)  # Replicates the test's f(a, b) function

def my_model_function():
    return MyModel()

def GetInput():
    # Generates input tensor matching the test's input shape (2,4) with indices in [0,9]
    return torch.randint(0, 10, (2, 4), dtype=torch.long)

