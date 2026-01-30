import torch
import scipy.linalg
arg_1_tensor = torch.rand([5, 5, 5], dtype=torch.complex128)
noise = torch.rand(arg_1_tensor.shape)
mask = noise < 0.05
noise_tensor = arg_1_tensor.clone()
noise_tensor[mask] = 0
noise_tensor[~mask] = 255
arg_1_tensor = noise_tensor
arg_1 = arg_1_tensor.clone()
print('scipy results')
print(scipy.linalg.expm(arg_1))
results = torch.matrix_exp(input=arg_1,)
print('torch results')
print(results)
# scipy results as: tensor([[[inf+nanj inf+nanj inf+nanj inf+nanj inf+nanj]...
# torch results as: tensor([[[inf+0.j, inf+0.j, inf+0.j, inf+0.j, inf+0.j]...