import random

import numpy as np
import torch

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)
tensor = torch.rand(10)
print(tensor.sum())
print(torch.dropout(tensor, p=0.5, train=True).sum())