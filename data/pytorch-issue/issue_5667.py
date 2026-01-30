m = nn.GRU(10,10).cuda()
data = Variable(torch.randn(10,10,10)).cuda()
m(data)

import torch.nn as nn
l = nn.LSTM(10, 10).cuda()