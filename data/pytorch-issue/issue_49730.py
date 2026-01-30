class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        input = input.contiguous()

        count = torch.empty(1,
                            dtype=running_mean.dtype,
                            device=input.device).fill_(input.numel() // input.size(1))

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.bn = nn.SyncBatchNorm(3, track_running_stats=False)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        return x


def run(rank, size):
    print(f"Running basic DDP example on rank {rank}.")
    model = MyNet().cuda().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    outputs = ddp_model(torch.randn(1, 3, 10, 10))
    pass


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()