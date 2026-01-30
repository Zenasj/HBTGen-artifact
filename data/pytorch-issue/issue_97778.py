import torch.nn as nn

fx_graph_runnable.py

NameError

math.floor

math_floor

math.ceil

math_ceil

import torch

def f(a, b, c):
    d = (torch.matmul(a, b) + c) / 2
    d_s0 = d.shape[0]
    d_s1 = d.shape[1]
    d_s3 = math.floor(d_s0 * d_s1)
    e = d.view(d_s3)
    return torch.cat([e, e])


import torch._export
from torch._export import export, dynamic_dim

inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
constraints = [
    dynamic_dim(inputs[0], 0),
    dynamic_dim(inputs[2], 0),
    dynamic_dim(inputs[2], 0) == dynamic_dim(inputs[0], 0),
]
gm = export(f, inputs, constraints, _add_runtime_assertions=False).graph_module
graph_module.to_folder("debug")

import torch
from math import inf
from math import nan
NoneType = type(None)
import torch
from torch import device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

from torch.nn import *
class FxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.load_state_dict(torch.load(r'debug/state_dict.pt'))

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        mm = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg1_1 = None
        add = torch.ops.aten.add.Tensor(mm, arg2_1);  mm = arg2_1 = None
        div = torch.ops.aten.div.Tensor(add, 2);  add = None
        sym_size = torch.ops.aten.sym_size(arg0_1, 0);  arg0_1 = None
        mul = sym_size * 7;  sym_size = None
        floor = math_floor(mul);  mul = None
        view = torch.ops.aten.view.default(div, [floor]);  div = floor = None
        cat = torch.ops.aten.cat.default([view, view]);  view = None
        return (cat,)

import torch
import math

@torch.compile(dynamic=True)
def f(a, b, c):
    d = (torch.matmul(a, b) + c) / 2
    d_s0 = d.shape[0]
    d_s1 = d.shape[1]
    d_s3 = math.floor(d_s0 * d_s1)
    e = d.view(d_s3)
    return torch.cat([e, e])

inputs = (torch.randn(2, 4), torch.randn(4, 7), torch.randn(2, 7))
f(*inputs)