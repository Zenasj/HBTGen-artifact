import numpy as np
import torch
print(torch.from_numpy(np.empty((0, 20))).shape)
print(torch.from_numpy(np.empty((20, 0))).shape)