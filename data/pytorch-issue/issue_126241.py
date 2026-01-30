import torch.nn as nn

import torch
from torch import nn
import torch._inductor.config as inductor_config
import torch._dynamo.config as dynamo_config

inductor_config.force_layout_optimization = True
dynamo_config.inline_inbuilt_nn_modules = True

torch.set_default_device("cuda")
torch.set_default_dtype(torch.bfloat16)

conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
x = torch.randn(4, 3, 224, 224)

@torch.compile
def f(x):
    x = conv1(x)
    x = conv2(x)
    return x

for i in range(2):
    f(x)
print("bye")

import torch

torch.set_default_device("cuda")

weight = torch.randn(64, 64, 1, 1)
x = torch.randn(2, 64, 10, 10).to(memory_format=torch.channels_last)
print(f"x stride {x.stride()=}")
print(f"before: {weight.stride()=}")
torch.convolution(x, weight, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
print(f"after: {weight.stride()=}")