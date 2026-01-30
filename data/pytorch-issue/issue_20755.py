import torch

a = torch.rand(3, 3, dtype = torch.float64)
print(a.dtype, a.device) # torch.float64 cpu
c = a.to(torch.float32)
#works

b = torch.load('bug.pt')
print(b.dtype, b.device) # torch.float64 cpu
c = b.to(torch.float32)
# RuntimeError: expected scalar type Float but found Double

d = b.clone().to(torch.float32)
# works