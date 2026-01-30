import numpy as np
import random

import torch
data = torch.tensor(np.random.randn(1, 10), dtype=torch.float32).to("cuda:0")
newout = data[range(data.shape[1]), 0]

import torch
data = torch.tensor(np.random.randn(1, 10), dtype=torch.float32).to("cpu")
newout = data[range(data.shape[1]), 0]