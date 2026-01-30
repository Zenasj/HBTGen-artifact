import torch
x1 = torch.randn(1, 72250, 1, device="mps")
x2 = torch.randn(1, 1, 72250, device="mps")
res = torch.bmm(x1, x2)