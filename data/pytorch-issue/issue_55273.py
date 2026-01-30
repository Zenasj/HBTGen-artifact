import torch.nn as nn

import torch
import numpy as np
import torch.nn.functional as F
import torch
import numpy as np

a = np.arange(1, 5).reshape((2, 2, 1)).astype(np.float32)
b = np.zeros((2, 2, 3)).astype(np.float32)
c = a + b

a1 = np.arange(1, 7).reshape((2, 1, 3)).astype(np.float32)
c1 = a1 + b

input_x1 = torch.tensor(a)
print("===========input_x1=========== \n", input_x1)
input_x2 = torch.tensor(a1)
print("===========input_x2=========== \n", input_x2)
cos = torch.nn.CosineSimilarity(2, eps=1e-8)
res = cos(input_x1, input_x2)
print("=========pytorch_broadcast_calculate\n", res)


input_x1_1 = torch.tensor(c)
print("===========input_x1========== \n", input_x1_1)
input_x2_1 = torch.tensor(c1)
print("===========input_x2========== \n", input_x2_1)
cos = torch.nn.CosineSimilarity(2, eps=1e-8)
res = cos(input_x1_1, input_x2_1)
print("=========pytorc_direct_calculate\n", res)