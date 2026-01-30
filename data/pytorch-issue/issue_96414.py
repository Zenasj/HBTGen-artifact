import torch.nn as nn

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerWithEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        layer = TransformerEncoderLayer(4, 1, 16)
        self.encoder = TransformerEncoder(layer, 1)
        self.embedding = nn.Embedding(16, 4, padding_idx=0)
    
    def forward(self, input):
        embedded = self.embedding(input)
        return self.encoder(embedded)

network = TransformerWithEmbedding().eval().cuda()
compiled_network = torch.compile(network, dynamic=True)
input = torch.tensor([[1, 2, 3, 4]], device="cuda:0", dtype=torch.int)
compiled_network(input)