python
import torch

x = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
y = torch.prod(x, dtype=torch.bool)

print(y)
# Get `tensor(True)` as result but the result should be `tensor(False)` obviously.