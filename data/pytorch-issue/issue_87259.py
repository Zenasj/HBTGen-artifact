import torch
size = 100000
a  = torch.randn(size)
b = torch.randn(3)
for i in range(10):
    c = torch.zeros((1, ))
    order = torch.randperm(size)
    for j in range(size):
        c += (a[order[j]] * b[2])
    print(c.item())