import torch.nn as nn

import torch

def compute(input, input2):
    clone = torch.ops.aten.clone.default(input2, memory_format = torch.channels_last)
    convolution = torch.ops.aten.convolution.default(clone, input, None, [2, 13], [1, 14], [1, 1], False, [0, 0], 67)
    convolution_backward = torch.ops.aten.convolution_backward.default(convolution, clone, input, [0], [2, 13], [1, 14], [1, 1], False, [0, 0], 67, [True, True, False])
    print(convolution_backward)

input = torch.randn(134, 67, 2, 1)
input2 = torch.randn(18, 4489, 23, 12)
input = input.contiguous(memory_format=torch.channels_last)

compute(input, input2)

import torch

input = torch.randn(18, 4489, 23, 12)
weight = torch.randn(134, 67, 2, 1)
weight = weight.contiguous(memory_format=torch.channels_last)
input_clone = torch.clone(input, memory_format=torch.channels_last)
convolution = torch.nn.Conv2d(4489, 134, (2, 1), stride=(2, 13), padding=(1, 14), dilation=(1, 1), bias=False, groups=67)
output = convolution(input_clone)