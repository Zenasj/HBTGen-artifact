import torch.nn as nn
import torch.nn.functional as F

from torch import nn
import torch
from torch.nn import functional as F


device = torch.device("mps")
model = nn.Sequential(
    nn.Linear(10, 1),
    nn.Sigmoid()
)
model.to(device)
input = torch.randn(5, 10).to(device)
target = torch.randint(0, 2, (5, 1)).float().to(device)

F.binary_cross_entropy(
    model(input),
    target
)