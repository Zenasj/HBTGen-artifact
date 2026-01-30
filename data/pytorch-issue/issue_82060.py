import torch.nn as nn

import torch

input = torch.randn(1, 1, 100, 100)
conv = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)

with torch.no_grad():
    out_ref = conv(input)

conv.to(memory_format=torch.channels_last)
with torch.no_grad():
    out = conv(input)

print(torch.mean(torch.abs(out - out_ref)))

import torch

torch.manual_seed(0)
input = torch.randn(1, 1, 100, 100)
def test():
	tmp_result= torch.nn.LazyConv2d(out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False)
	return tmp_result
conv = test()

with torch.no_grad():
    out_ref = conv(input)

conv.to(memory_format=torch.channels_last)
with torch.no_grad():
    out = conv(input)

result=torch.mean(torch.abs(out - out_ref))

print(result)
# tensor(0.6507)