import torch.nn as nn

import torch

with torch.inference_mode():
    d_model = 4
    layer = torch.nn.TransformerEncoderLayer(d_model, 2, 2, batch_first=True)
    layer.eval()
    x = torch.randn(5, 10, d_model)
    pad = torch.rand(5, 10) > 0.5
    layer(x, src_key_padding_mask=pad)