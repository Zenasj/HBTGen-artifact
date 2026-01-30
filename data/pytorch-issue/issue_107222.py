import random

import torch
import numpy as np
input_np=np.random.random((40000, 5)).astype(np.float64)
x_py=torch.from_numpy(input_np.astype(input_np.dtype)).cuda()

torch.geqrf(x_py)  # got error here

b1,b2=torch.geqrf(x_py)

print(b1)  # got error here

torch.geqrf(x_py)