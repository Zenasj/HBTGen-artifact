import torch.nn as nn

import torch
from torch import nn
@torch.no_grad()
def run_conv(device, mod):
    conv = nn.Conv2d(1, 1, kernel_size=3).to(device)
    conv.bias -= 10    # to ensure negative activations
    act = conv(torch.zeros((1,3,3)).to(device))
    print(mod(act),'\t',repr(mod) )
run_conv('cpu', nn.ReLU(inplace=True))
run_conv('mps', nn.ReLU(inplace=True))
run_conv('mps', nn.ReLU(inplace=False))
run_conv('mps', lambda x: nn.ReLU(inplace=True)(x*1.0))