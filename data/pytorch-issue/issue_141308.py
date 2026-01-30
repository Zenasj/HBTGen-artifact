import torch.nn as nn

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

x1 = torch.ones(2,3)
y1 = torch.ones(2,3)
z1 = torch.ops.aten.set_.source_Tensor(x1, y1)

fake_tensor_mode = FakeTensorMode()
x2 = fake_tensor_mode.from_tensor(torch.ones(2,3))
y2 = fake_tensor_mode.from_tensor(torch.ones(2,3))
z2 = torch.ops.aten.set_.source_Tensor(x2, y2)

print(f"x1: {x1.untyped_storage()._cdata}, y1: {y1.untyped_storage()._cdata}, z1: {z1.untyped_storage()._cdata}")
print(f"x2: {x2.untyped_storage()._cdata}, y2: {y2.untyped_storage()._cdata}, z2: {z2.untyped_storage()._cdata}")
# x1: 99973024, y1: 99973024, z1: 99973024
# x2: 112107232, y2: 112107232, z2: 112107232

import torch

def fn(x):
    p = torch.nn.Parameter(x + 123)
    return p, p.sin()

opt = torch.compile(fn, fullgraph=True)
x = torch.ones(16, device="cuda", requires_grad=True)

p, r = opt(x)
r.sum().backward()

import torch

x = torch.ones(2,3)
x_detach = x.detach()
y = torch.ones(2,3)
z = torch.ops.aten.set_.source_Tensor(x_detach, y)


print(f"x: {x.untyped_storage()._cdata}, x_detach: {x_detach.untyped_storage()._cdata}, y: {y.untyped_storage()._cdata}, z: {z.untyped_storage()._cdata}")
# x: 97023632, x_detach: 97026480, y: 97026480, z: 97026480

def forward(self, primals_1: "f32[16][1]cuda:0", primals_2: "f32[16][1]cuda:0"):
   # File: /home/boyuan/playground/inductor/donated_buffer.py:4 in fn, code: p = torch.nn.Parameter(x + 123)
  add: "f32[16][1]cuda:0" = torch.ops.aten.add.Tensor(primals_1, 123);  primals_1 = None
  
   # File: /home/boyuan/playground/inductor/donated_buffer.py:5 in fn, code: return p, p.sin()
  sin: "f32[16][1]cuda:0" = torch.ops.aten.sin.default(add)
  
  # No stacktrace found for following nodes
  set_: "f32[16][1]cuda:0" = torch.ops.aten.set_.source_Tensor(primals_2, add);  primals_2 = set_ = None
  return (sin, add)