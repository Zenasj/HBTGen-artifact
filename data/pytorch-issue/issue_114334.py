import torch.nn as nn

import torch
i = torch.randn(1,1)
t = torch.randn(1,1,1,1,1)
m = torch.nn.AdaptiveLogSoftmaxWithLoss(1,3,(1,),div_value=0.0)
result = m(i,t)