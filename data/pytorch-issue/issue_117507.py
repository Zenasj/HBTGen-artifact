import torch
a = torch.rand(1, 3, 2)
b = torch.rand(1, 3, 2500*2500) # 1 x 3 x 6250000
# no error
torch.linalg.lstsq(a, b)
# error
torch.linalg.lstsq(a.cuda(), b.cuda())
# no error if making b smaller
torch.linalg.lstsq(a.cuda(), b[:, :, :65536].cuda())

import torch
print (torch.__version__)
print (torch.version.cuda)
print (torch.cuda.get_device_properties (0))

_ = torch.manual_seed (2024)

a = torch.rand(1, 3, 2)
b = torch.rand(1, 3, 2500*2500) # 1 x 3 x 6250000
torch.linalg.lstsq(a.cuda(), b.cuda())