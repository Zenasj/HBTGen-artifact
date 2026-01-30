import torch
import logging

# torch._logging.set_logs(all=logging.DEBUG)

@torch.compile
def my_model(x):
    y = torch.matmul(x, x).cuda()
    y = torch.matmul(y, x).cuda()
    return y

x = torch.ones(10, 10).cuda()

y = my_model(x)

input("`lsof | grep nvidia` and found a lot of sub-processes also obtained the GPU device")