import torch.nn as nn

import torch
y = torch.rand((1, 1, 3, 32, 32))
y_pad = torch.nn.functional.pad(y, (5, 5, 5, 5, 5, 5), mode='replicate')

import torch
y = torch.rand((1, 1, 3, 32, 32), device='mps')
y_pad = torch.nn.functional.pad(y, (5, 5, 5, 5, 5, 5), mode='replicate')