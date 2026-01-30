import torch.nn as nn

py
import torch


batch_size = 32
seqlen = 20
d_model = 64
nhead = 4

encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, bias=False, batch_first=True).eval()
x = torch.randn(batch_size, seqlen, d_model)

y = encoder_layer(x)  # AttributeError: 'NoneType' object has no attribute 'device'

tensor_args = (
              src,
              self.self_attn.in_proj_weight,
              self.self_attn.in_proj_bias,
              self.self_attn.out_proj.weight,
              self.self_attn.out_proj.bias,
              self.norm1.weight,
              self.norm1.bias,
              self.norm2.weight,
              self.norm2.bias,
              self.linear1.weight,
              self.linear1.bias,
              self.linear2.weight,
              self.linear2.bias,
          )