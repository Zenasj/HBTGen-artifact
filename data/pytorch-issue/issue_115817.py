import torch.nn as nn

import torch
from torch.fx.experimental.proxy_tensor import make_fx

class OneHot(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return torch.nn.functional.one_hot(input, num_classes=5)

onehot = OneHot()
input = torch.arange(0, 5)
fx_g = make_fx(onehot)(input)

def one_hot(tensor, num_classes):
    index = torch.arange(0, num_classes, device=tensor.device)
    return (
        tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [num_classes])
    ).to(torch.int64)