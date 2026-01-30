import torch
a = torch.ones(2, 2, requires_grad=True);
b = torch.randn(2, 2);
c = a + b;
print(c)
c.backward();