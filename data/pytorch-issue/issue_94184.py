import torch

x = torch.tensor([[1.0]], requires_grad=True)
print(x.shape)
y = x.var()
y.backward()
print(x.grad)

import torch

x = torch.zeros([1, 1, 1], requires_grad=True) 
y = x.var(unbiased=True) # unbiased=False is fine, no divide by zero
print("var(x): ", y) # if one sample, unbiased variance is nan (undefined); if more than one sample (e.g., torch.ones([1, 2]), unbiased variance is 0
y.backward()
print("d var(x)/dx: ", x.grad) # if one sample, grad is nan; if more than one sample, grad is 0