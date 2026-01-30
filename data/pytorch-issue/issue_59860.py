import torch
x = torch.rand(4000, 20, 10, 20, dtype=torch.float64)
x_cu = x.cuda()

for dim in range(x.ndim):
    y = torch.cumsum(x, dim=dim)
    y_cu = torch.cumsum(x_cu, dim=dim)
    err = torch.sum(torch.abs(y_cu.cpu() - y))
    print(dim, err)