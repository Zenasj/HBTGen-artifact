import torch
a = torch.zeros(1).cuda(1)

import torch
a = torch.zeros(1).cuda(1)

#1  0x00007fff622869eb in at::CUDAGenerator::CUDAGenerator(at::Context*) () from /data/users/sgross/pytorch/torch/lib/libATen.so
#2  0x00007fff62286d98 in at::Context::doInitCUDA() () from /data/users/sgross/pytorch/torch/lib/libATen.so