# torch.rand(B, seq_len, embedding_dim, dtype=torch.float32)  # Inferred input shape: (batch_size, seq_len, embedding_dim)
import math
import torch
from torch import nn, Tensor

class MyModel(nn.Module):
    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def my_model_function():
    # Returns PositionalEncoding model with default parameters
    return MyModel()

def GetInput():
    # Generate random input tensor matching [batch_size, seq_len, embedding_dim]
    batch_size = 32
    seq_len = 10
    embedding_dim = 512
    return torch.rand(batch_size, seq_len, embedding_dim, dtype=torch.float32)

