import numpy as np
import torch

W = torch.tensor([[4.,0], [0 ,3.]])
print(f'Torch matrix norm: {W.norm(2, (0,1))}, Numpy matrix norm: {np.linalg.norm(W.numpy(), 2)}')