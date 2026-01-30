import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

        def forward(self, x):
            return F.linear(x, self.weight)