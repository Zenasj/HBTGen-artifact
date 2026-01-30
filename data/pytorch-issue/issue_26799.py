import torch
divide_tensor = torch.Tensor([1.0, 0.0, 1.0])
weight = torch.ones(3)
weight.requires_grad = True
tmp = weight / divide_tensor
loss = tmp[0]
loss.backward()
print(weight.grad)

import torch
divide_tensor = torch.Tensor([1.0, 0.0, 1.0])
weight = torch.ones(3)
weight.requires_grad = True
tmp = weight[0] / divide_tensor[0]
loss.backward()
print(weight.grad)