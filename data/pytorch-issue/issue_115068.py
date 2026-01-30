import random

import torch
import numpy as np
x = torch.tensor(np.random.rand(), dtype=torch.float16)
y = torch.tensor(np.random.rand(0), dtype=torch.float16)
torch.true_divide(x, y)