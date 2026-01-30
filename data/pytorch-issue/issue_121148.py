import random

import torch
import numpy as np
input = torch.tensor(27)
other = torch.tensor(np.random.rand(6,0), dtype=torch.half)
torch.inner(torch.tensor(27), torch.tensor(np.random.rand(6,0), dtype=torch.half))