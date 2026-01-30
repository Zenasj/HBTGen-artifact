import torch.nn as nn

import torch
from torch import nn

model = nn.MultiheadAttention(640, 1, batch_first=True)
query = torch.randn(1, 1, 640)
key = torch.randn(4, 196, 640)
value = torch.randn(4, 196, 640)
out, weights = model(query, key, value)
print(out.shape, weights.shape)