import torch
import numpy as np

a_n = np.zeros([10])
a_t = torch.from_numpy(a_n)
torch.ones([], dtype=a_t.dtype).set_(a_t.storage(), 0, (20,))