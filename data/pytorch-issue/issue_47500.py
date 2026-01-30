import torch
import numpy as np

x = torch.tensor(0. - 1.0000e+20j, dtype=dtype, device=device)
self.compare_with_numpy(torch.sqrt, np.sqrt, x)

x = torch.tensor(-1.0000e+20 - 4988429.2000j, dtype=dtype, device=device)
self.compare_with_numpy(torch.sqrt, np.sqrt, x)