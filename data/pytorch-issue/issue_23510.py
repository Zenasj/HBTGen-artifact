import torch

x = torch.tensor(1.0, dtype=torch.double, requires_grad=True).expand(2, 3)
source = torch.zeros(2, 2, dtype=torch.double)
index = torch.randint_like(source, 3, dtype=torch.long)
gradcheck(lambda x: x.scatter_(1, index, source), x)

x = torch.zeros(5, 6, dtype=torch.double, requires_grad=True)
source = torch.zeros(10, 10, dtype=torch.double, requires_grad=True)
index = torch.randint(5, (2, 6), dtype=torch.long)
x.scatter(0, index, source)
print('forward success')
x.scatter(0, index, source).sum().backward()
print('backward success')

x = torch.zeros(5, 6, dtype=torch.double, requires_grad=True)
source = torch.zeros(10, 10, dtype=torch.double, requires_grad=True)
index = torch.randint(5, (2, 2), dtype=torch.long)
x.scatter(0, index, source)
print('forward success')