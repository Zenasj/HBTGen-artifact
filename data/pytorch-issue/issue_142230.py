import torch.nn as nn

import torch

def fn(x):
    softmax = torch.nn.functional.softmax(x, dim=-1)
    sum = torch.sum(softmax, dim=-1)
    sum_broadcast = torch.broadcast_to(
        sum.unsqueeze(-1), [*(sum.size()[0:3]), 256]
    )
    sum_exp = torch.exp(sum_broadcast)
    return torch.sum(sum_exp, dim=-1)

x = torch.randn(4, 12, 1023, 1022)
foo = torch.compile(fn)
foo(x)