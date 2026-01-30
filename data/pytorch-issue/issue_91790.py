import torch
import numpy as np

print('torch:', torch.__version__)

print(np.array(0+0j)**0)
print(torch.tensor(0+0j)**0)
print(torch.tensor(0+0j)**torch.tensor(0))