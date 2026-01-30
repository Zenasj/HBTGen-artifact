import torch.nn as nn

import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        s1 = max(x.shape[0], y.shape[0])
        s2 = max(x.shape[1], y.shape[1])
        z = torch.zeros((s1, s2), dtype=x.dtype)
        z[:x.shape[0], :x.shape[1]] = x
        z[:y.shape[0], :y.shape[1]] += y
        return z


model = Model()
x = torch.arange(6).reshape((2,3))
y = torch.arange(6).reshape((3,2)) * 10
z = model(x, y)
print(f"x.shape={x.shape}, y.shape={y.shape}, z.shape={z.shape}")


DYN = torch.export.Dim.DYNAMIC

ep = torch.export.export(model, (x,y), dynamic_shapes=({0:DYN, 1:DYN},{0:DYN, 1:DYN}))
print(ep)

# %%
# But does it really work?
# We just print the shapes.
model_ep = ep.module()
ez = model_ep(x,y)
print("case 1:", z.shape, ez.shape)

x = torch.arange(4).reshape((2,2))
y = torch.arange(9).reshape((3,3))
try:
    ez = model_ep(x,y)
    print("case 2:", model(x,y).shape, ez.shape)
except Exception as e:
    print("case 2 failed:", e)