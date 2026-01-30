import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
encoder = nn.TransformerEncoder(
    encoder_layer, num_layers=6, enable_nested_tensor=True)
encoder.cuda()

x = torch.randn(1, 1026, 512)
x_mask = [[0]*1025+[1]*1]
x_mask = torch.Tensor(x_mask).bool()

x = x.cuda()
x_mask = x_mask.cuda()

encoder.eval()
with torch.no_grad():
    y = encoder(x, src_key_padding_mask=x_mask)
print(y.shape)

import torch
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048, batch_first=True)
encoder = nn.TransformerEncoder(
    encoder_layer, num_layers=6, enable_nested_tensor=True)
encoder.cuda()

x = torch.randn(1, 1026, 512)
x_mask = [[0]*1024+[1]*2]     # Here I change the number of tokens to 1024
x_mask = torch.Tensor(x_mask).bool()

x = x.cuda()
x_mask = x_mask.cuda()

encoder.eval()
with torch.no_grad():
    y = encoder(x, src_key_padding_mask=x_mask)
print(y.shape)  # torch.Size([1, 1024, 512])