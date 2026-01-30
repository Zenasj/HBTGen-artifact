import torch
x = torch.tensor([1., 2.], dtype=torch.float32)
print("float32 : ", torch.einsum('i, i -> ', x, x)) 

y = torch.tensor([1., 2.], dtype=torch.float64)
print("float64 : ", torch.einsum('i, i -> ', y, y)) 

z = torch.tensor([1, 2])
print("int : ", torch.einsum('i, i -> ', z, z))