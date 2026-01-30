import numpy as np

import torch; 
p = torch.tensor([0.2, 0.9]) 
n = int(2000e4);
x = torch.rand(n);
torch.quantile(x, p)

torch.quantile

[0, 1]

np.percentile

[0, 100]

np.quantile