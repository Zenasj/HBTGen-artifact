import torch
import numpy as np

# select elements according to flag
flag1 = torch.Tensor([0, 1, 0, 1, 0])
flag2 = np.array([0, 1, 0, 1, 0])
values = torch.arange(len(flag1))

selected1 = values[flag1 == 1]
selected2 = values[flag2 == 1]
selected3 = values.numpy()[flag2 == 1]

print(selected1)  # tensor([1, 3])
print(selected2)  # tensor([0, 1, 0, 1, 0])
print(selected3)  # [1, 3]