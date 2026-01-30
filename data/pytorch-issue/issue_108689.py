import torch
import numpy as np
inds = np.zeros((16,16,16))
x = torch.ones(16,16,16)
x[inds <= 0]

import torch
import numpy as np
inds = np.zeros((32,32,32))
x = torch.ones(32,32,32)
x[inds <= 0]