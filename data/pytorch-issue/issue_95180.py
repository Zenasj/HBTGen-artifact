import torch

base = torch.tensor(0.00, requires_grad=True)
x = torch.tensor([0.00], requires_grad=True)
res = torch.pow(base, x)
order = 2
for i in range(order):
    res, = torch.autograd.grad(res, (x, ), create_graph=True)
    print(f"{i+1}-order gradient with respect to base: {res}")