import torch.nn as nn

import torch
# for 3d
torch.nn.functional.adaptive_avg_pool3d(torch.randn([2,2,2,2]), []) 
# for 2d
torch.nn.functional.adaptive_avg_pool2d(torch.randn([2,2,2,2]), [])