import os
import torch
import torch.nn as nn

import torch.distributed as dist
import torch.multiprocessing as mp


NUM_GPUS = 2


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.w * x.pow(2) + self.b * x/x  # !!! 


def worker(rank):

    torch.manual_seed(rank)
    torch.cuda.set_device(rank)
    device = torch.device(rank)
    dist.init_process_group(backend='nccl', world_size=NUM_GPUS, rank=rank)

    def parallelize(module):
        return nn.parallel.DistributedDataParallel(module, device_ids=[rank])

    model = Model().to(device)
    model = parallelize(model)

    # all workers have the same initial model
    w = model.module.w
    b = model.module.b
    print(f'initial weights at {rank}:', w.data, b.data)

    x = torch.randn(3).to(device)
    x.requires_grad = True
    y = model(x)  # shape [3]

    # all workers have different data
    print(f'input data at {rank}:', x)

    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    loss = grad.pow(2).mean(0)  # gradient penalty

    # compare with gradient calculated by hand
    assert torch.isclose(2 * x * w, grad).all()

    model.zero_grad()
    loss.backward()

    # all workers have the same grad
    print(f'final gradient at {rank}:', w.grad, b.grad)

    # compare with gradient calculated by hand
    t = (8 * x.pow(2) * w).mean(0)
    print(f'local gradient at {rank}:', t)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    assert torch.isclose(t/NUM_GPUS, w.grad).all()


def main():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(worker, nprocs=NUM_GPUS, args=())


if __name__ == '__main__':
    main()

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.w * x.pow(2) + self.b  # x/x is removed here!