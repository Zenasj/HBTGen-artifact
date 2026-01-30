import random

import torch
import numpy as np
row = np.random.randint(3, 6)
col = np.random.randint(3, 6)
offset = np.random.randint((- 1), 2)
torch_offset = torch.triu_indices(row, col, offset=offset, dtype=torch.float)