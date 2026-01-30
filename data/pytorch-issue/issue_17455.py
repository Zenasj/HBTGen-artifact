import torch
import numpy as np

x = torch.from_numpy(np.array([0, 100])).float()
x.requires_grad = True
f = torch.log((torch.exp(x) + 1))

f[1] = 0
f[:] = 0

clean_f = torch.zeros(100)
clean_f[0] = f[0]

print(torch.autograd.grad(f[1], x, retain_graph=True))
print(torch.autograd.grad(f[0], x, retain_graph=True))
print(torch.autograd.grad(clean_f[1], x, retain_graph=True))
print(f)
print(torch.__version__)