import torch
t = torch.rand(2, 3).cuda()
r = t.type(torch.cuda.FloatTensor)

import torch
t = torch.rand(2, 3).privateuseone()
r = t.type(torch.privateuseone.FloatTensor)