import torch
b = torch.randn(600,1)
A = torch.randn(600,2)
X, _ = torch.gels(b,A)
X.size() #torch.Size([2, 1])
Y, _ = torch.gels(b.cuda(),A.cuda())
Y.size() #torch.Size([600, 1])