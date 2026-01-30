import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

model = spectral_norm(nn.Linear(3, 3)).cuda()
x = torch.zeros((4, 3)).cuda()

with torch.autocast(device_type="cuda"):
    model(x)