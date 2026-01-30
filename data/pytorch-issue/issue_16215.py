import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import functional as F

def print_max_diff(x, y):
    print((x - y).abs().max())

def have_fun(x, w):
    z_linear = F.linear(x.permute(0, 2, 1), w).permute(0, 2, 1)
    z_conv_f = F.conv1d(x, w.unsqueeze(2))
    
    layer = nn.Conv1d(2, 2, 1, bias=False)
    sd = layer.state_dict()
    sd["weight"] = w.unsqueeze(2)
    layer.load_state_dict(sd)
    z_conv = layer(x)
    
    print(z_linear)
    print(z_conv_f)
    print(z_conv)
    print_max_diff(z_linear, z_conv)
    print_max_diff(z_linear, z_conv_f)

x = torch.rand(1, 2, 3)
w = torch.rand(2, 2)

have_fun(x, w)

have_fun(x, w.inverse())

have_fun(x, w.inverse().clone())

have_fun(x, w.t())

have_fun(x, w.t().clone())