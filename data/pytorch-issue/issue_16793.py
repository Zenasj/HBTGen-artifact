import torch.nn as nn

import torch 
bn=torch.nn.BatchNorm2d(3) 
print(bn.running_mean) 
#tensor([0., 0., 0.])
bn.track_running_stats = False 
for i in range (10): 
    bn(torch.rand(1, 3, 2, 2)) 
print(bn.running_mean)
#tensor([0.32, 0.12, 0.12])