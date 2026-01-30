import torch
torch.manual_seed(420)
torch.cuda.manual_seed_all(420)
x = torch.randn(10, 10).log() # x contains NaN
y = torch.histc(x, bins=10, min=0, max=1)
print(y) # first element is [48, ...] which counts all the NAN elements

x = x.cuda()
y = torch.histc(x, bins=10, min=0, max=1)
print(y) # first element is only [2, ...] which does not count any NAN elements

print(torch.isnan(x.view(-1)).sum().item()) # amount of NAN is 46 = 48 -2

tensor([48.,  0.,  2.,  3.,  1.,  3.,  0.,  2.,  0.,  1.])
tensor([2., 0., 2., 3., 1., 3., 0., 2., 0., 1.], device='cuda:0')
46