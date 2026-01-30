import torch
import torch.nn as nn

class Repro(torch.nn.Module):
    def __init__(self):
        super(Repro, self).__init__()
        self.register_buffer('foo', torch.empty(2,3))
        self.register_buffer('bar', torch.empty(4,5))
r = Repro()
expected = r.get_buffer('bar')