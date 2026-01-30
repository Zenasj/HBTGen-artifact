import torch

slize = [1, 2, 3, 4]
x = torch.randn(10, requires_grad=True)
y = x[slize]
y.sum().backward()
# breaks second time calling backward
y.sum().backward() # rasises RuntimeErroe

x = torch.randn(10, requires_grad=True)
y = x[1:4]
y.sum().backward()
y.sum().backward()