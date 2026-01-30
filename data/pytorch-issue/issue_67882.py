import torch.nn as nn

# $ torchrun --standalone --nnodes=1 --nproc_per_node=1 script.py
import os
rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

import torch.distributed
# !!!!!! Uncomment the following and the script succeeds
# torch.distributed.rpc.init_rpc('worker', rank=rank, world_size=world_size)

import torch.distributed as dist
dist.init_process_group(backend='gloo')

import torchvision.models as models
import torch

rn50 = models.resnet50()
rn50.train()
rn50 = torch.nn.parallel.DistributedDataParallel(rn50)

from torch.distributed.rpc import RRef
from torch.distributed.optim import DistributedOptimizer

params = []
for param in rn50.parameters():
    params.append(RRef(param))


dist_optim = DistributedOptimizer(
        torch.optim.SGD,
        params,
        lr=0.05)

loss_func = torch.nn.CrossEntropyLoss()

with torch.distributed.autograd.context() as context_id:
    pred = rn50(torch.randn(50, 3, 224, 224))
    target = torch.randn(50, 1000).softmax(dim=1)
    loss = loss_func(pred, target)
    dist.autograd.backward(context_id, [loss])
    dist_optim.step(context_id)