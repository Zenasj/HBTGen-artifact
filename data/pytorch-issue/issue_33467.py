import torch

a = torch.zeros((2, 1, 0))
b = torch.zeros((2, 0, 1))
c = torch.ones((2, 1, 1))
torch.bmm(a, b) + c     # <-- ones((2, 1, 1))  (OK)
torch.baddbmm(c, a, b)  # <-- zeros((2, 1, 1)) (WRONG)