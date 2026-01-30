import torch

f = torch.randn(30,18)
u,s,v = torch.svd(f,some=False)
print('u shape:',u.shape)  #u shape: torch.Size([30, 30])
print('s shape:',s.shape)   #s shape: torch.Size([18])
print('v shape:',v.shape)   #30634 v shape: torch.Size([18, 18])