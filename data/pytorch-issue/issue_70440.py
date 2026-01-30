import torch
import torch.nn as nn
layer = nn.ConvTranspose3d(
            in_channels=768, out_channels=768,
            kernel_size=[2, 1, 1], dilation=[7, 1, 1], output_padding=0)
layer_norm = nn.LayerNorm(768, eps=0.1)

t = torch.ones(2, 768, 1, 3, 3)
t = layer(t)
t = t.permute(0, 2, 3, 4, 1).contiguous()
loss = layer_norm(t.contiguous())
loss.sum().backward()