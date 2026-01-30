import torch

a=torch.linspace(10000,1.7,10000)
b=a.cuda()
a.sum()
# tensor(50008496.)
b.sum()
# tensor(50008504., device='cuda:0')