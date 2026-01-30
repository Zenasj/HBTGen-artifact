import torch
import torch.nn as nn


class Repro(nn.Module):
    def forward(self, x):
        return x.var_mean()

torch.compile(Repro())(torch.rand(100))