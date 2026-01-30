import torch

x = torch.ones(1)
x.requires_grad_()
a = 1
f = torch.zeros(10)
f[0] = a
for i in range(1,10):
    f[i] = f[i-1]*x
    f[i] = 1/(1.0+torch.exp(-f[i]))
loss = torch.sum(f)
grad = torch.autograd.grad(loss,[x])[0]

x= torch.ones(1)
x.requires_grad_()
a = 1
f = torch.zeros(10)
f[0] = a
for i in range(1,10):
    tmp = a
    for j in range(i):
        tmp = 1/(1.0+torch.exp(-(a*x)))
    f[i] = tmp
loss = torch.sum(f)
grad = torch.autograd.grad(loss,[x])[0]