import torch.nn as nn
import random

import numpy as np
import torch
params = torch.tensor(np.random.rand(4,0), dtype=torch.half)
ids = torch.tensor([[0,51,81]])
# tensor([], size=(4, 0), dtype=torch.float16) tensor([[ 0, 51, 81]])
max_norm = -0.8705713102734194
res = torch.nn.functional.embedding(ids, params, max_norm=max_norm)