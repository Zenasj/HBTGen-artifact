import torch
import math
import torch.nn as nn
from torch.nn import Transformer

# torch.rand(10, 2, 512, dtype=torch.float)

class PositionalEncoding(nn.Module):
    r"""Inject positional encodings into input sequence."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MyModel(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, max_len=5000):
        super(MyModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.transformer = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)

    def forward(self, src):
        src = self.pos_encoder(src)
        # Use Transformer's encoder component (common in encoder-only setups)
        return self.transformer.encoder(src)

def my_model_function():
    # Returns a model with default parameters (d_model=512, nhead=8, etc.)
    return MyModel()

def GetInput():
    # Generate input tensor matching (seq_len=10, batch_size=2, d_model=512)
    return torch.rand(10, 2, 512, dtype=torch.float)

