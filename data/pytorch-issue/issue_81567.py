import torch
a=torch.tensor([9.0,3.0, 5.0, 4.0])
a=a.to(torch.device('mps'))
a.type(torch.LongTensor)

tensor([         9,          3, 8675683992,          0])

tensor([         0,          0, 4335857328, 4506956464])

tensor([                   0,                    0, -8070450532247928832, -8070450532247928832])

tensor([0, 0, 0, 0])

import torch
a=torch.tensor([9.0,3.0, 5.0, 4.0])
a=a.type(torch.LongTensor)
a.to(torch.device('mps'))

tensor([9, 3, 5, 4], device='mps:0')