import torch.nn as nn

import torch

device = torch.device('mps')
t = torch.randn(1).to(device)
act = torch.nn.Softplus().to(device)
o = act(t)