import torch
import torch.nn as nn

t = torch.zeros((1, 100, 100))
t = torch.nn.Upsample(scale_factor=2)(t) # warning here
t = torch.nn.Upsample(scale_factor=2.5)(t) # no warning