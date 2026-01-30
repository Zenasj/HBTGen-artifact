import torch.nn as nn

import torch
import torch.nn.functional as F

test = torch.ones((64), device='mps')
testPadded = F.pad(test,(0,1),value=0.)