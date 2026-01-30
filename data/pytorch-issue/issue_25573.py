import torch

a = torch.arange(10)
print(a)
a[1:] = a[:9]
print(a)

tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tensor([0, 0, 1, 2, 3, 4, 5, 6, 7, 8])

tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tensor([0, 0, 1, 2, 3, 4, 5, 6, 7, 7])

import numpy as np
a = np.arange(10)
print(a) # [0 1 2 3 4 5 6 7 8 9]
a[1:] = a[:9]
print(a) # [0 0 1 2 3 4 5 6 7 8]