import torch
import torch.nn as nn

torch.fx.wrap("wrapped")

def wrapped(x: torch.Tensor, r: torch.nn.ReLU):
    return r(x)

class M(torch.nn.Module):

    def __init__(self):
        super(M, self).__init__()
        self.r = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        res = wrapped(x, self.r)
        return res

"""
Throws “NotImplementedError: argument of type: <class 'torch.nn.modules.activation.ReLU’>”
"""