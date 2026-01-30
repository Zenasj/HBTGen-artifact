import random

import numpy as np
import torch


np.random.seed(0)
na = np.random.randn(256, 256).astype(np.float32)
nb = np.random.randn(256, 256).astype(np.float32)
nres = na @ nb  #  nres = na.dot(nb),  nres = na.__matmul__(nb)
print(nres.sum())  # >>> out: -2984.1348


ta = torch.from_numpy(na)
tb = torch.from_numpy(nb)
tres = ta @ tb  # tres = ta.matmul(tb)
print(tres.sum())  # >>> out: tensor(-2984.1350)