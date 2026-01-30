import torch
import torch.nn as nn

net = torch.nn.DataParallel(net).cuda()