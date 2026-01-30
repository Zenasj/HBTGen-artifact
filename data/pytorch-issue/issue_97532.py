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
    encoder_layer, 2, encoder_norm, enable_nested_tensor=True
).to('cuda')

inputs = torch.randn(2,3,16).to('cuda')
inputs.requires_grad=False

src_mask = torch.zeros(3, 3, dtype=torch.bool)#.triu_(diagonal=1)
input_seq_len = torch.tensor([3,2])
padding_mask = (torch.arange(3)[None, :].cpu() >= input_seq_len[:, None])

assert(src_mask.dtype == padding_mask.dtype)
assert(src_mask.dtype == torch.bool)
      
encoder.eval()

with torch.no_grad():
  out = encoder(inputs, 
      mask=src_mask.cuda(),
      src_key_padding_mask=padding_mask.cuda(),
  )