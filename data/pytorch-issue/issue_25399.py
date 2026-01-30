import torch.nn as nn

# test.py
import torch

x = torch.zeros(10, 3)
p = torch.nn.Parameter(x)
p = torch.nn.Parameter(data=x)  # second try