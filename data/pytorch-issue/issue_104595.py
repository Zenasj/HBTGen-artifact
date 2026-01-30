import torch.nn as nn

import torch
from torch import nn

transformer_model = nn.Transformer(
    nhead=1, 
    d_model=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    batch_first=True,
    dropout=0.0, 
)
src = torch.rand((1, 10, 4))
tgt = torch.rand((1, 20, 4))

transformer_model.eval()

out = transformer_model(
    src, 
    tgt,
    tgt_mask=transformer_model.generate_square_subsequent_mask(tgt.size(1)),
)

out_limited = transformer_model(
    src, 
    tgt[:, :10, :],
    tgt_mask=transformer_model.generate_square_subsequent_mask(tgt[:, :10, :].size(1)),
)

print((out[:, :10]-out_limited).mean())