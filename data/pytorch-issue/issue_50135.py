import torch
import torch.nn as nn

# no error
input = torch.randn(1, 1, 5, 5)
pool = nn.MaxPool2d(4, 4, padding=2)
print(pool(input).shape)
# >>> torch.Size([1, 1, 2, 2])

# it raises error
input = torch.randn(1, 1, 5, 5)
pool = nn.MaxPool2d(4, 4, padding=3)
print(pool(input).shape)