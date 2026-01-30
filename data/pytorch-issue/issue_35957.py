import torch.nn as nn
import random

#!/usr/bin/env python3

import os, sys
import numpy as np
import torch

cuda_flag = 0

def numpy_to_torch_var(np_t):
    if cuda_flag: t = torch.from_numpy(np_t).cuda()
    else: t = torch.from_numpy(np_t).clone()
    t.requires_grad = True
    return t

class SomeFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, in_act):
        return in_act

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.clone()

prev_layer = numpy_to_torch_var(np.random.randn(1, 84, 84, 4).astype(np.float32))
prev_layer = torch.transpose(prev_layer, 1, 3).contiguous()
weight_var = numpy_to_torch_var(np.random.randn(4, 4, 1, 1).astype(np.float32))
prev_layer = torch.nn.functional.conv_transpose2d(prev_layer, weight_var, None)
prev_layer = torch.transpose(prev_layer, 1, 3).contiguous()
prev_layer = SomeFunc.apply(prev_layer)

print("Should fail here")
prev_layer.sum().backward()
print("Should have failed here")