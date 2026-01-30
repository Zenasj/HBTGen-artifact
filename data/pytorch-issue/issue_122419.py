import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        t = torch.tensor(x.size(-1), device=self.device, dtype=torch.float)
        t = torch.sqrt(t * 3)
        return x * t