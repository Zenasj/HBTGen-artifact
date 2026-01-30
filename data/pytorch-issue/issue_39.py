import torch
import torch.nn as nn

bn = nn.BatchNorm1d(100)
input = Variable(torch.randn(10,100).cuda())

bn(input) # produce std::bad_cast

bn.cuda()
bn(input) # works fine