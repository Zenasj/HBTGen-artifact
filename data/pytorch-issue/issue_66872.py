import torch.nn as nn

import torch
def cpu():
  arg_0 = torch.rand(torch.Size([500, 6]), dtype=torch.float32)
  arg_1 = torch.randint(-32768,32768,torch.Size([500]), dtype=torch.int64)
  res = torch.nn.functional.nll_loss(arg_0,arg_1,)
  
def cuda():
  arg_0 = torch.rand(torch.Size([500, 6]), dtype=torch.float32).cuda()
  arg_1 = torch.randint(-32768,32768,torch.Size([500]), dtype=torch.int64).cuda()
  res = torch.nn.functional.nll_loss(arg_0,arg_1,)
  
print(cuda())
print("CUDA PASS!")
print(cpu())