import torch.nn as nn

import torch
from torch.nn.functional import grid_sample
import numpy as np

a = np.zeros((1,2,10,10), dtype=np.float32)
for i in range(10):
    for j in range(10):
        a[0,0,i,j] = j
        a[0,1,i,j] = i
at = torch.from_numpy(a)
pos = at.permute(0,2,3,1)  # should be pixel index, used when align_corners=False

# case 1: grid is the pixel center, align_corners=False, output: 2.3842e-06
print((grid_sample(at, (pos+0.5)/10*2-1, align_corners=False)-at).abs().sum())
# case 2: grid is the pixel index, align_corners=True, output: 3.5763e-06
print((grid_sample(at, pos/9*2-1, align_corners=True)-at).abs().sum())
# case 3: grid is the pixel center, align_corners=True, output: 50.0000
print((grid_sample(at, (pos+0.5)/10*2-1, align_corners=True)-at).abs().sum())
# case 4: grid is the pixel index, align_corners=False, output: 199.4445
print((grid_sample(at, pos/9*2-1, align_corners=False)-at).abs().sum())

# test gpu kernel, result is similar
print((grid_sample(at.cuda(), (pos.cuda()+0.5)/10*2-1, align_corners=False)-at.cuda()).abs().sum())
print((grid_sample(at.cuda(), pos.cuda()/9*2-1, align_corners=True)-at.cuda()).abs().sum())
print((grid_sample(at.cuda(), (pos.cuda()+0.5)/10*2-1, align_corners=True)-at.cuda()).abs().sum())
print((grid_sample(at.cuda(), pos.cuda()/9*2-1, align_corners=False)-at.cuda()).abs().sum())