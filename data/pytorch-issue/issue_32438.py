import torch
import numpy as np
a, b = torch.tensor(0.), torch.tensor(np.nan)
print('min(0,nan): ', torch.min(a, b) )
print('min(nan, 0): ', torch.min(b, a))