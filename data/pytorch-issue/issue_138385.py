import torch.nn as nn

import numpy as np
import torch

input = torch.tensor(np.nan, dtype=torch.float64)
lambd = torch.tensor(1.0, dtype=torch.float64)

out = torch.nn.functional.softshrink(input,lambd)
print(out) # tensor(0., dtype=torch.float64) actual; NaN expected.