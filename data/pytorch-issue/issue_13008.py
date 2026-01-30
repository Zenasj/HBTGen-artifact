import torch.nn as nn

import torch
import torch.nn.functional as F
input = torch.randn(1,1,5,5, requires_grad=True)
weight = torch.randn(1,1,3,3, requires_grad=True)
output = F.conv2d(input, weight, dilation=2)
grad_output = torch.randn(output.shape)
grad_input = torch.autograd.grad(output, input, grad_output)
torch.nn.grad.conv2d_input(input.shape, weight, grad_output, dilation=2)

def dim_size(d):
     return ((grad_output.size(d + 2) - 1) * stride[d] - 2 * padding[d] + 
                kernel_size[d] + (kernel_size[d] - 1) *  (dilation[d] - 1))