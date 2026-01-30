import torch

a = torch.tensor([1., 2.], requires_grad=True).clone()
a[0] = a[1].sin() 
a.sum().backward()