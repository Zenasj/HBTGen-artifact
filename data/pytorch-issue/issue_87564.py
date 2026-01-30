import torch

device = "meta"

pred = torch.randn(5, 5, device=device) > 0
a = torch.rand(5, 5, device=device).t()
out = torch.where(pred, a, 0)

print("pred.stride()", pred.stride())
print("a.stride()", a.stride())
print("out.stride()", out.stride()) 

pred.stride() (5, 1)
a.stride() (1, 5)
out.stride() (1, 5)

pred.stride() (5, 1)
a.stride() (1, 5)
out.stride() (5, 1)