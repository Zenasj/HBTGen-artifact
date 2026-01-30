import torch.nn as nn

import torch

print(torch.__version__)

try:
    torch.nn.LayerNorm(normalized_shape=(3,3))(torch.empty(3,3, dtype=torch.float64))
except Exception as e:
    print(repr(e))