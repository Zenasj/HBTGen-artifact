import torch
import torch_xla

import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch._dynamo as dynamo


@dynamo.optimize("torchxla_trace_once")
def fn_simple(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    return a + b


x = torch.tensor(100.0)
y = torch.tensor(200.0)
res = fn_simple(x, y)

import torch
import torch_xla

import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch._dynamo as dynamo



@dynamo.optimize("torchxla_trace_once")
def fn_fallback(M, mat1, mat2, beta):
  # xla currently only support alpha and beta == 1
  return torch.addmm(M, mat1, mat2, beta=beta)


M = torch.randn(2, 3)
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)

res = fn_fallback(M, mat1, mat2, 0.5)