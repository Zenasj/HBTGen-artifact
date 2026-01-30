py
import torch
from torch import Tensor
from typing import *


import torch

@torch.library.custom_op("_reinplacing::add_one", mutates_args={"result"})
def add_one(x: torch.Tensor, result: torch.Tensor) -> None:
    result.copy_(x + 1)

factory_op = torch.zeros_like

class AddOne(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = factory_op(x)
        add_one(x, out)
        ctx.save_for_backward(out)
        return out
 
    @staticmethod
    def backward(ctx, grad):
        saved, = ctx.saved_tensors
        out = factory_op(grad)
        add_one(saved, out)
        return out

@torch.compile(backend="inductor")
def f(x):
    return AddOne.apply(x)

x = torch.randn(3, requires_grad=True, device="cuda")
y = f(x)