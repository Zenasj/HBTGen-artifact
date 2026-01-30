import torch

x = torch.Tensor([1,0,2,3])
y = torch.Tensor([2,0,0,0])
k = torch.BoolTensor([True, False, True, True])
x.requires_grad = True
out = torch.atan2(y[k], x[k])
out.mean().backward()
print(x.grad)
# -> tensor([-0.1333,  0.0000,  0.0000,  0.0000])

x = torch.Tensor([1,0,2,3])
y = torch.Tensor([2,0,0,0])
k = torch.BoolTensor([True, False, True, True])
x.requires_grad = True
out = torch.atan2(y, x)
out[k].mean().backward()
print(x.grad)
# -> tensor([-0.1333,     nan, -0.0000, -0.0000])