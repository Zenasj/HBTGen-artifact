import random

import numpy as np
np.random.seed(1234)
import torch

input = torch.tensor(4, dtype=torch.float32)
dim = 0
index = torch.tensor(19, dtype=torch.int64)
tensor2 = torch.tensor(np.random.randint(0, 50, (1,3,4,4)), dtype=torch.float32)