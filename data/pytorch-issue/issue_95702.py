import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=16,
    nhead=2,
    dim_feedforward=32,
    dropout=0.1,
    activation='relu',
    batch_first=True,
)
encoder_norm = nn.LayerNorm(16)
encoder = nn.TransformerEncoder(
    encoder_layer, 2, encoder_norm
)

inputs = torch.randn(2,3,16)

src_mask = torch.ones(3, 3, dtype=torch.bool).triu_(diagonal=1)
input_seq_len = torch.tensor([3,2])
padding_mask = (torch.arange(3)[None, :].cpu() >= input_seq_len[:, None])

assert(src_mask.dtype == padding_mask.dtype)

encoder(inputs, 
    mask=src_mask,
    src_key_padding_mask=padding_mask,
)