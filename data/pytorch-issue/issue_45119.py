import torch
import torch.nn as nn

x = nn.Parameter(torch.rand(10))
bounds = torch.arange(5) / 5.
torch.bucketize(x, bounds)