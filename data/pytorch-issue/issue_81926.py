import torch
a= torch.tensor([1]).cuda()
print(torch.arange(2048, device = a.device, dtype=torch.half).long())
print(torch.arange(2048, device = a.device, dtype=torch.long))

tensor([   0,    1,    2,  ..., 2045, 2046, 2047], device='cuda:0')
tensor([   0,    1,    2,  ..., 2045, 2046, 2047], device='cuda:0')

import torch
a= torch.tensor([1]).cuda()
print(torch.arange(4096, device = a.device, dtype=torch.half).long())
print(torch.arange(4096, device = a.device, dtype=torch.long))

tensor([   0,    1,    2,  ..., 4092, 4094, 4096], device='cuda:0') #Not expected!
tensor([   0,    1,    2,  ..., 4093, 4094, 4095], device='cuda:0')