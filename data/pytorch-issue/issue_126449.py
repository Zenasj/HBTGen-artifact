import torch
cfn = torch.compile(torch.constant_pad_nd)
cfn(torch.zeros(5, dtype=torch.bool), [2, 3])