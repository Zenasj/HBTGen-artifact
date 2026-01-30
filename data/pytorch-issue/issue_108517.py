import torch.nn as nn

import torch
mha = torch.nn.MultiheadAttention(100, 4)
x = torch.random(2, 10, 100)
mha(x, x, x, is_causal=True, need_weights=False)  # RuntimeError