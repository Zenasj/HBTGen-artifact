import torch.nn as nn

import torch

print(torch.__version__)

a = torch.randint(10, size=(3, 3))
a = a - 5
print(a)
print(torch.nn.functional.relu(a))

b = torch.Tensor([1, -1])
print(b)
print(torch.nn.functional.relu(b))