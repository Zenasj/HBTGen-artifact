import torch
t = torch.randn(())
index = torch.tensor([0])
source = torch.randn(1,1,1)
t.index_add(0,index,source)