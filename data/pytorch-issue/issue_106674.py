import torch

x = torch.tensor([[[0, 1]]], dtype=torch.uint8)

y0 = torch.ops.aten.upsample_nearest1d.default(x, [257])
y1 = torch._decomp.decompositions.upsample_nearest1d(x, [257])

assert (y0 == y1).all()