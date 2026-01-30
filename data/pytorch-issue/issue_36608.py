import torch

root = torch.arange(9.).reshape(3, 3).requires_grad_()
x = root.clone()
v1 = x.unbind()
v2 = v1[0].narrow(0, 0, 2)
v2.mul_(2)  # errors out as it should
v2.sum().backward()
print(root.grad)