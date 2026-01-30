import torch

print('---------torch.det----------')


x1 = torch.zeros(2,1, dtype=torch.float, requires_grad=True)
x2 = torch.ones(2,1, dtype=torch.float, requires_grad=True)
x = torch.hstack((x1,x2))
x.retain_grad()
y = torch.det(x)
y.backward()
print('input matrix:')
print(x)
print('gradients on input matrix:')
print(x.grad)


print('---------torch..eig---------')


w1 = torch.zeros(2,1, dtype=torch.float, requires_grad=True)
w2 = torch.ones(2,1, dtype=torch.float, requires_grad=True)
w = torch.hstack((w1,w2))
w.retain_grad()
z = torch.prod(torch.linalg.eigvals(w))
z.backward()
print('input matrix:')
print(w)
print('gradients on input matrix:')
print(w.grad)