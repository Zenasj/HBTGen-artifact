import torch
import torch.nn as nn

conv1 = nn.Conv2d(3, 128, kernel_size=1, bias=False)
bn = nn.BatchNorm2d(128)
relu1 = nn.ReLU(inplace = True)
relu2 = nn.ReLU(inplace = True)

input = torch.randn(10,3,8,8)
x = conv1(input)
x = bn(x)
x = relu1(x)

# the second successive inplace ReLU operation
x = relu2(x)

x = x.flatten()
x = torch.sum(x) 
x.backward()