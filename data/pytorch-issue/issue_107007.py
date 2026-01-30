import torch

x = torch.tensor([1,2,3,4], requires_grad=True, dtype=torch.float32)
y = x - x.mean() # y = [1,2,3,4] - 2.5 = [-1.5, -0.5, 0.5, 1.5]
y.backward(torch.ones_like(y))
print(x.grad) # x.grad should be 1-1/4=[0.75, 0.75, 0.75, 0.75] but it is [0, 0, 0, 0]