import torch
import numpy as np 

a = torch.tensor([-1e+19+1e+19j], dtype=torch.complex64, device='cuda')
print('torch asinh', a.asinh()) # will output torch asinh tensor([-inf+0.j], device='cuda:0')


x = np.array(a.cpu())
print('numpy asinh', np.arcsinh(x)) # will output numpy asinh [-44.788837+0.7853982j]