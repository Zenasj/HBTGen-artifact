import torch

self = torch.full((9, 1, 1, 9, 1, 8, 8, 7, 8,), 1.4013e-45, dtype=torch.float)
padding = [0, 0, 0, 0, 0, 0]
torch.ops.aten.replication_pad3d(self, padding)