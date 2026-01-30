import random

import torch
import numpy as np

input = torch.tensor(np.random.rand(14,7))
input2 = torch.tensor(np.random.rand(12,6))
torch.orgqr(input, input2)

import torch
import numpy as np

input = torch.tensor(np.random.rand(14,7))
input2 = torch.tensor(np.random.rand(12,16)) # changed here
torch.orgqr(input, input2)