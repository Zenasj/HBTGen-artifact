# torch.rand(B, S, E, dtype=torch.float32)  # Example shape: (20, 5, 10)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, embedding_dim=10):
        super(MyModel, self).__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)  # Fixed variable name per documentation fix

    def forward(self, x):
        return self.layer_norm(x)

def my_model_function():
    # Uses default embedding dimension from the example (10)
    return MyModel()

def GetInput():
    B, S, E = 20, 5, 10  # Batch, sentence length, embedding dimension
    return torch.randn(B, S, E, dtype=torch.float32)  # Matches the issue's example

