import torch
t = torch.rand(4, 3, 1, 2)
m = torch.ones((4, 3), dtype=torch.bool)
t[m, [0]].shape # (12, 1)
t.numpy()[m.numpy(), [0]].shape # (12, 2)