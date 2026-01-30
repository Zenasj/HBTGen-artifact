import torch.nn as nn

import torch

input1 = torch.randint(high=(1 << 7) - 1, size=[1], dtype=torch.int8)
input2 = torch.randn(size=[1], dtype=torch.bfloat16)

module = torch.nn.Bilinear(in1_features=1,in2_features=1,out_features=0,bias=True,dtype=torch.complex64)
output = module(input1, input2)

import torch

input1 = torch.randint(high=(1 << 7) - 1, size=[1], dtype=torch.int64)
input2 = torch.randn(size=[1], dtype=torch.float32)

module = torch.nn.Bilinear(in1_features=1,in2_features=1,out_features=0,bias=True,dtype=torch.float32)
output = module(input1, input2)