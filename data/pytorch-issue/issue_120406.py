import torch.nn as nn

import torch
import torch.nn.functional as F
a = F.conv3d(torch.ones(2, 3, 8, 9, 26), torch.ones(3, 1, 1, 1, 17), groups=3)