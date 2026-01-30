import numpy as np
import torch

a = torch.tensor([2], dtype=torch.float16)
b = torch.tensor(np.array([], dtype='float16'))
print(torch.mul(a, b))
# print(torch.div(a, b))