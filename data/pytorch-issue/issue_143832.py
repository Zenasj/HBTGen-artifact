import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention

torch.compiler.reset()
flex_attention = torch.compile(flex_attention)
torch.manual_seed(1)
x = torch.rand(1, 1, 32, 256).to(device="cuda")
flex_attention(x, x, x)