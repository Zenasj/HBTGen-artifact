import torch.nn as nn

import torch
import torch.nn.functional as F                                                                                                                                                                                                               
x = torch.rand(200, 24, 56, 56, dtype=torch.float16, device='cuda').to(memory_format=torch.channels_last) 
w = torch.rand(24, 1, 3, 3, dtype=torch.float16, device='cuda').to(memory_format=torch.channels_last) 
r = F.conv2d(x, w, None, (2,2), (1, 1), (1, 1), 24)