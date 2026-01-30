import torch 
from torch import distributions as tdist

m = tdist.MultivariateNormal(torch.zeros([170,128,3,3]).cuda(), (torch.eye(3)*3).cuda())

m = tdist.MultivariateNormal(torch.zeros([170,128,4,3]).cuda(), (torch.eye(3)*3).cuda())
m = tdist.MultivariateNormal(torch.zeros([171,128,3,3]).cuda(), (torch.eye(3)*3).cuda())