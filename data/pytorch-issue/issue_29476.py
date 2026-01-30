import torch

print(T)
print (type(T))
print (Dirichlet(1/T-1 * torch.ones(T-1)).sample([10]))
print (Dirichlet(1/29 * torch.ones(29)).sample([10]))