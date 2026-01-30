import torch
arg_0 = torch.rand(torch.Size([16, 264, 32, 32]), dtype=torch.float32)
arg_1 = torch.rand(torch.Size([20, 2048]), dtype=torch.float32)
res = torch.matmul(arg_0,arg_1,)

import torch
arg_0 = torch.rand(torch.Size([16, 264, 32, 32]), dtype=torch.float32).cuda()
arg_1 = torch.rand(torch.Size([20, 2048]), dtype=torch.float32).cuda()
res = torch.matmul(arg_0,arg_1,)