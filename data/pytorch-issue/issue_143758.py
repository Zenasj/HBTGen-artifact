import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import functional as F


lin_layer = nn.Linear(200, 300)
input = torch.randn(1, 400, 200)
out_lin_layer = lin_layer(input)
out_lin = F.linear(input, lin_layer.weight.data, lin_layer.bias.data)
out_manual = input @ lin_layer.weight.data.t() + lin_layer.bias.data

print(f"torch.allclose(out_lin_layer, out_lin): {torch.allclose(out_lin_layer, out_lin)}") # prints True
print(f"torch.allclose(out_lin_layer, out_manual): {torch.allclose(out_lin_layer, out_manual)}")  # prints False

import torch
from torch import nn
from torch.nn import functional as F


lin_layer = nn.Linear(200, 300, bias=False)
input = torch.randn(1, 400, 200)
out_lin_layer = lin_layer(input)
out_lin = F.linear(input, lin_layer.weight.data)
out_manual = input @ lin_layer.weight.data.t()

print(f"torch.allclose(out_lin_layer, out_lin): {torch.allclose(out_lin_layer, out_lin)}") # prints True
print(f"torch.allclose(out_lin_layer, out_manual): {torch.allclose(out_lin_layer, out_manual)}")  # prints True