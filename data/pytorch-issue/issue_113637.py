import torch
import torch.nn as nn

class Model0(torch.nn.Module):
    def forward(self, *args):
        _args = args
        getitem = _args[0]
        max_1 = getitem.max(1)
        getattr_1 = max_1.values
        argmin = getattr_1.argmin(1)
        neg = torch.neg(getattr_1)
        to = neg.to(dtype = torch.int32)
        return (argmin, to)