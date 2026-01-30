import torch.nn as nn
input_size = [1,2]
m = nn.CrossMapLRN2d(1,0.1,0.1,1)
m(torch.rand(input_size))