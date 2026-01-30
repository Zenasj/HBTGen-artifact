import torch

input = torch.rand(5, 4, 16, 16, dtype=torch.complex64)
input.requires_grad_()
weight = torch.rand(8, 4, 3, 3, dtype=torch.complex64)
torch.nn.functional.conv2d(input, weight)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, bias:bool=True, contiguous:bool=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.contiguous = contiguous
        self.weight = nn.Parameter(torch.complex(torch.Tensor(out_features, in_features),
                                                 torch.Tensor(out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.complex(torch.Tensor(out_features),
                                                   torch.Tensor(out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return _apply_linear_complex(F.linear, x, self.weight, self.bias, self.contiguous)


def _apply_linear_complex(linear_fn, x, weight, bias, contiguous=True):
    x_r, x_i = x.real, x.imag
    w_r, w_i = weight.real, weight.imag
    b_r, b_i = (None, None) if bias is None else (bias.real, bias.imag)
    y_rr = linear_fn(x_r, w_r, b_r)
    y_ir = linear_fn(x_i, w_r, b_r)
    y_ri = linear_fn(x_r, w_i, b_i)
    y_ii = linear_fn(x_i, w_i, b_i)
    return torch.complex(y_rr-y_ii, y_ir+y_ri)