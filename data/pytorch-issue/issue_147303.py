import torch.nn as nn

import torch


op = torch.nn.BatchNorm2d(num_features=3, affine=True, track_running_stats=True)
ifm = torch.empty(size=[16, 3, 224, 224]).uniform_(0, 1).to(dtype=torch.float16)
ifm = ifm.contiguous(memory_format=torch.channels_last) # this creates Nan in output
ifm = ifm.requires_grad_()

res = op(ifm)
bwd_tensor = torch.empty(size=res.shape).uniform_(0, 1).to(dtype=torch.float16)

res.backward(bwd_tensor)

print(ifm.grad)