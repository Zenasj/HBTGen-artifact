import random

import torch
import numpy as np
np.random.seed(2023)
x = torch.tensor(np.random.randint(0, 100, ()), dtype=torch.uint8)
print(x)  # tensor(87, dtype=torch.uint8)
out = torch.square(x)
print(out)  # tensor(145, dtype=torch.uint8)