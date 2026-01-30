import torch.nn as nn

import torch
from torch import nn
from torch._subclasses import FakeTensorMode
mode = FakeTensorMode()
large_tensor = torch.randn(64, 16, 32, 32, device="meta")
large_tensor = torch._subclasses.FakeTensor(fake_mode=mode, elem=large_tensor, device="cuda:0")
net = nn.ConvTranspose2d(16, 16, 6).to(device="meta")
for name, p in net.named_parameters():
    fake_p = torch.nn.Parameter(torch._subclasses.FakeTensor(fake_mode=mode, elem=p, device="cuda:0"))
    setattr(net, name, fake_p)
y = net._conv_forward(large_tensor, net.weight, net.bias)
print(type(y))

weight_on_the_fly = conv.weight
bias_on_the_fly = conv.bias
weight_on_the_fly, bias_on_the_fly = fold_bn_into_weights(weight_on_the_fly, bias_on_the_fly, bn_module)
output = conv. _conv_forward(input, weight_on_the_fly, bias_on_the_fly)