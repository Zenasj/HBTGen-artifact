import torch

A = torch.arange(9, dtype=torch.float32).view(1, 3, 3)
    
def func(A):
    output_tensor = A.clone()
    index = output_tensor > 1e-9
    output_tensor[index] = 1e-09
    return output_tensor

vjp = jacrev(func)(A)