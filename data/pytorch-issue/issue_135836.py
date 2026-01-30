import numpy as np
import torch
import torch.nn as nn

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
y = torch.from_numpy(x)