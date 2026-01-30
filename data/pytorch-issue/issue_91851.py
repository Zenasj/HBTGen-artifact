import torch
a = torch.tensor(5.0, requires_grad=True) * 0.1
b = torch.tensor(2.0, requires_grad=True)
c = a + b
c.backward()

print(a.grad, b.grad)

import torch
a = torch.tensor(5.0) * 0.1
a.requires_grad = True
b = torch.tensor(2.0, requires_grad=True)
c = a + b
c.backward()
print(a.grad, b.grad)