import torch

zz = torch.ones(1, 2, 3).cuda()
for i in range(3):
    print(zz[0, [i]])
    print(zz.cpu()[0, [i]])

zz = torch.ones(1, 2, 3).cuda()
zz[0, [2]]