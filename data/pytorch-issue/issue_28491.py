import torch
import torch.nn as nn

class X(object):
    pass

class Y(X):
    pass

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.y = Y()

    def forward(self):
        return self.y