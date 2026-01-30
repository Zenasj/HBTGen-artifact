import torch

b = torch.randn(3, requires_grad=True)
c = torch.zeros(3) 
c[[1,2]] = b[[1,1]]
c.sum().backward(retain_graph=True) # ok
c.sum().backward() # ok

b = torch.randn(3, requires_grad=True)
c = b * 2
c.sum().backward() # ok
c.sum().backward() # fail