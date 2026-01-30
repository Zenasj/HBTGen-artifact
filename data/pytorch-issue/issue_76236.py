import torch
torch.nonzero(torch.tensor(0))
# output tensor([], size=(0, 0), dtype=torch.int64)
torch.nonzero(torch.tensor(1))
# output tensor([], size=(1, 0), dtype=torch.int64)

import numpy as np
np.array(np.nonzero(0))
# array([], shape=(1, 0), dtype=int64)
np.array(np.nonzero(1))
# array([[0]]), shape (1, 1)