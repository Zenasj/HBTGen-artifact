import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from typing_extensions import reveal_type

def register_hook(module: DistributedDataParallel, state: object, hook: Callable):
    reveal_type(module.register_comm_hook) # This outputs Union[Tensor, Module]
    module.register_comm_hook(state, hook) # This throws a typing error: "Tensor" not callable

import math
import torch
from torch import nn, Tensor
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)