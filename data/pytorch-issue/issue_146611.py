import random

import torch
import numpy as np
x1 = torch.tensor(np.random.randn(10, 10))
x2 = torch.tensor(np.random.randn(10))
vec1 = torch.tensor(np.random.randn(100, 3, 4))
vec2 = torch.tensor(np.random.randn(100, 4, 5))
vec3 = torch.tensor(np.random.randn(3, 4))
vec4 = torch.tensor(np.random.randn(4, 5))
vec5 = torch.tensor(np.random.randn(4))
## Below shows 4 usage in 4 funcs with: beta==0 && input of unexpected size
out1 = torch.addbmm(x1, vec1, vec2, beta=0) # (1) torch.addbmm()
# out2 = torch.baddbmm(x1, vec1, vec2, beta=0) # (2) torch.baddbmm()
# out3 = torch.addmm(x1, vec3, vec4, beta=0) # (3) torch.addmm()
# out4 = torch.addmv(x2, vec3, vec5, beta=0) # (4) torch.addmv()