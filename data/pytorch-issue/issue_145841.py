import numpy as np
import torch

file = np.load("tensor.zip")
tensor = torch.from_numpy(file.f.arr_0)
print(tensor)

print(tensor.shape)

print(torch.ge(tensor.clamp_min(1e-6), 0.0).all())
print(torch.ge(tensor.abs()+0.01, 0.0).all())
# print(torch.ge(tensor.abs()+0.01, 0.0).all())