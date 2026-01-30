import torch

t = torch.range(1,10)
t[1:] = t[:-1] # wrong result
print(t)
t = torch.range(1,10)
t[1:] = t[:-1].clone() # right result
print(t)
t = torch.range(1, 10).numpy()
t[1:] = t[:-1] # right result
print(t)