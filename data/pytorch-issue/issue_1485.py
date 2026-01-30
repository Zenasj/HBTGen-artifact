import torch
import torch.nn as nn
ec=nn.Conv3d(1,1,3,2,1).cuda()
input=torch.autograd.Variable(torch.ones(1,1,511,512,512),requires_grad=True).cuda()
ec(input).sum().backward()