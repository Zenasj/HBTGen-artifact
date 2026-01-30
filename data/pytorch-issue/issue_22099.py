import numpy as np
import torch

arr = np.array([None] * 10000000)
while True:
    try:
        torch.as_tensor(arr.copy(), device=None)
    except TypeError:
        pass