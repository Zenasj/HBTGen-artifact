import torch
a = torch.tensor((1.0,))
print(a.new_ones((1,), requires_grad=True).requires_grad)
print(a.new_zeros((1,), requires_grad=True).requires_grad)