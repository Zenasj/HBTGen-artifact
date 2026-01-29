# torch.rand(2, 10, 768, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, max_seq_length=512, embedding_dim=768):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(embedding_dim, max_seq_length))  # Positional encoding buffer

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:, :seq_len]  # Slice positional encoding based on input sequence length

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, 768, dtype=torch.float32)  # Batch=2, seq_len=10, embedding_dim=768

