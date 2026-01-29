# torch.rand(1, 1024, 32, 32, dtype=torch.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import logging

class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_):
        out = torch.sign(input_)
        ctx.save_for_backward(input_)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[input_.gt(1)] = 0
        grad_input[input_.lt(-1)] = 0
        return grad_input

class BiGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None):
        super(BiGroupNorm, self).__init__(num_groups, num_channels, eps, affine, device, dtype)
        self._logger = logging.getLogger()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        self.weight = Parameter(self.weight - self.weight.mean())
        self.weight = Parameter(BinaryQuantize().apply(self.weight))
        self.bias = Parameter(BinaryQuantize().apply(self.bias))
        
        self._logger.debug(f'weight shape of {self._get_name()}: {self.weight.shape}, sum of abs weight: {torch.sum(torch.abs(self.weight))}')
        self._logger.debug(f'bias shape of {self._get_name()}: {self.bias.shape}, sum of abs bias: {torch.sum(torch.abs(self.bias))}')
        
        return F.group_norm(input_, self.num_groups, self.weight, self.bias, self.eps)

class MyModel(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(MyModel, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=True)
        self.bi_group_norm = BiGroupNorm(num_groups, num_channels, eps=eps, affine=True)
    
    def forward(self, x):
        out1 = self.group_norm(x)
        out2 = self.bi_group_norm(x)
        return torch.mean(torch.abs(out1 - out2))

def my_model_function():
    return MyModel(num_groups=32, num_channels=1024)

def GetInput():
    return torch.rand(1, 1024, 32, 32, dtype=torch.float32)

