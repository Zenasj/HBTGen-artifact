import torch.nn as nn

from torch import nn
import torch
import time
from torch.amp import autocast
device = torch.device('cuda')

hidden_dim = 1024
USE_FP16 = True
BS = 2
TOTAL_INPUT = 250
decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim,
                                           dim_feedforward=hidden_dim*4,nhead=8,batch_first=True,)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3,).to(device)

with autocast('cuda', enabled=USE_FP16):
    with torch.no_grad():
        memory = torch.rand((1, 1000, hidden_dim), device=device)
        tgt = torch.rand((1, 36, hidden_dim), device=device)
        out = transformer_decoder(tgt, memory)
        s = time.perf_counter()
        for _ in range(TOTAL_INPUT):
            out = transformer_decoder(tgt, memory)
        print(time.perf_counter() - s)


        memory = torch.rand((BS, 1000, hidden_dim), device=device)
        tgt = torch.rand((BS, 36, hidden_dim), device=device)


        s = time.perf_counter()
        for _ in range(TOTAL_INPUT // BS):
            out = transformer_decoder(tgt, memory)
        print(time.perf_counter() - s)