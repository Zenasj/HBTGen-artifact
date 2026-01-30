import torch.nn as nn
import torch.nn.functional as F

import math

import torch
from torch.nn.modules.activation import MultiheadAttention

torch.manual_seed(42)
data = torch.randn(3, 2, 128) * math.sqrt(128)
data_0 = data[:, 0].unsqueeze(1)
att = MultiheadAttention(128, 8, 0.)
assert torch.allclose(att.forward(data, data, data)[0][:, 0], 
                      att.forward(data_0, data_0, data_0)[0][:, 0])

torch.allclose(att.forward(data, data, data)[0][:, 0], att.forward(data_0, data_0, data_0)[0][:, 0]) == True

import math

import torch
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

torch.manual_seed(42)
data = torch.randn(3, 2, 128) * math.sqrt(128)
data_0 = data[:, 0].unsqueeze(1)
att = MultiheadAttention(128, 8, 0.)
att.eval()

batched = F.linear(data, att.in_proj_weight, att.in_proj_bias)
single = F.linear(data_0, att.in_proj_weight, att.in_proj_bias)
assert torch.allclose(batched[:, 0], single[:, 0])