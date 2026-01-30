import numpy as np
import torch
a = np.array([1,2,3])
torch.from_numpy(a) # This works fine

b = a % 2
b

torch.from_numpy(b)

b.dtype , b.dtype.num

a.dtype, a.dtype.num