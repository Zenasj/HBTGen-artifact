import torch
x = torch.ones(0)
print(x.int().max().item()) # Gives -2147483648
x = x.cuda()
print(x.int().max().item()) # Gives 0

import torch
import numpy as np

print(np.ones((0, 3, 4)).max(1).shape) # (0, 4)
print(np.ones((0, 3, 4)).max(2).shape) # (0, 3)

print(*map(lambda x:x.size(), torch.ones((0, 3, 4)).max(1))) # (0, 4) (0, 4)
print(*map(lambda x:x.size(), torch.ones((0, 3, 4)).max(2))) # (0, 3) (0, 3)

print(np.ones((0, 3, 4)).max(0).shape) # Raises an identity error
print(*map(lambda x:x.size(), torch.ones((0, 3, 4)).max(0))) # Raises an identity error