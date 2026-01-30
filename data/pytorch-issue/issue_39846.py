import torch.nn as nn

import numpy as np
import torch

mp = torch.nn.MaxPool3d(kernel_size=2, stride=2, dilation=1)
empty = torch.tensor([np.nan]*(4**3)).reshape([1, 1, 4, 4, 4]).cuda()
print (mp(empty))

# tensor([[[[[-3.4028e+38, -3.4028e+38],
#            [-3.4028e+38, -3.4028e+38]],

#           [[-3.4028e+38, -3.4028e+38],
#            [-3.4028e+38, -3.4028e+38]]]]], device='cuda:0')

tensor([[[[nan, nan],
          [nan, nan]]]], device='cuda:0')