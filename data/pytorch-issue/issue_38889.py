import torch
x = torch.rand(10, dtype=torch.double)
normal = torch.distributions.Normal(0,1)
normal.icdf(x)

normal = torch.distributions.Normal(0,1)
normal.icdf(x.cuda())

normal = torch.distributions.Normal(0,1)
normal.icdf(x.cuda(1))

import torch

x = torch.rand(10, dtype=torch.double, device="cuda:1")
val = torch.tensor(1.)
y=x * val
torch.cuda.synchronize()
print("ok")
y=val * x
torch.cuda.synchronize()
print("ok") #not ok, illegal memory access

import torch
x = torch.rand(10, dtype=torch.double, device="cuda:1")
normal = torch.distributions.Normal(0,1)
with torch.cuda.device_of(x):
    normal.icdf(x)

torch.cuda.set_device(0)
x = torch.rand(10, dtype=torch.double, device="cuda:1")
val = torch.tensor(1.)
y = val * x
torch.cuda.synchronize()

torch.cuda.set_device(1)
x = torch.rand(10, dtype=torch.double, device="cuda:0")
val = torch.tensor(1.)
y = val * x
torch.cuda.synchronize()

torch.cuda.set_device(0)
x = torch.rand(10, dtype=torch.double, device="cuda:0")
val = torch.tensor(1.)
y = val * x
torch.cuda.synchronize()

torch.cuda.set_device(1)
x = torch.rand(10, dtype=torch.double, device="cuda:1")
val = torch.tensor(1.)
y = val * x
torch.cuda.synchronize()