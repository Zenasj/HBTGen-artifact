import torch

a = torch.tensor([0., 2.])
b = torch.ones((2), requires_grad=True)
c = torch.sum(a**b)
c.backward()
print('grad b', b.grad)    # is [nan, 1.386] should be [0, 1.386]