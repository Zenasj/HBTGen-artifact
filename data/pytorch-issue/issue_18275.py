import torch

matrix1 = torch.ones(32,32)
matrix1.requires_grad = True

nuc_norm = torch.norm(matrix1, p='nuc')

nuc_norm.backward() # works fine


tensor1 = torch.ones(10,3,32,32)
tensor1.requires_grad = True

nuc_norm = torch.norm(tensor1, p='nuc', dim=(2,3))

loss = nuc_norm.sum()

loss.backward() # U, V not computed error