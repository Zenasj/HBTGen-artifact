import torch.nn as nn
m = nn.InstanceNorm3d(16,16,1,False,True)
input_size = [0,16,16,16,16]
torch.compile(m)(torch.rand(input_size))