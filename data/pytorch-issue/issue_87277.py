import torch.nn as nn

import torch
y = torch.ones((50, 9, 300))
y_pad = torch.nn.functional.pad(y, (0, 0, 0, 31))

import torch
y = torch.ones((50, 9, 300), device='mps')
y_pad = torch.nn.functional.pad(y, (0, 0, 0, 31))