import torch.nn.functional as F

import torch
from torch.nn import functional as F
from torch.autograd import Function
class FunctionConvTranspose2d_(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return F.conv_transpose2d(input, weight, bias, stride, padding, dilation, groups)
x = torch.rand((1, 8, 32, 32)).cuda().requires_grad_()
weight = torch.rand(8, 8, 8, 1).cuda().requires_grad_()
bias = torch.rand(8).cuda().requires_grad_()
FunctionConvTranspose2d_.apply(x, weight, bias, 2)