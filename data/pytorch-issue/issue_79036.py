import torch
x = torch.rand(2, 2, 2, 2)
qx = torch.quantize_per_channel(x, scales=torch.tensor([0.5, 0.1]), zero_points=torch.tensor([102, 2]), axis=0, dtype=torch.qint8)
qx.min()