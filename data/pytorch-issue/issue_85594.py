import torch.nn as nn

import torch
max_pool = torch.nn.MaxPool1d(3)
t = torch.rand([17, 0, 50], dtype=torch.float32)  # note requires_grad is False
max_pool(t) # Worked and returned tensor of shape [17, 0, 48].

import torch
max_pool = torch.nn.MaxPool1d(3)
t = torch.rand([17, 0, 50], dtype=torch.float32)  # note requires_grad is False
max_pool(t) # Errors with `max_pool1d: Expected 2D or 3D (batch mode) tensor with optional 0 dim batch size for input, but got: [17, 0, 48]`