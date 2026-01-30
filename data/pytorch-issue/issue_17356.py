import os
import torch
import torch.nn as nn
from torch.multiprocessing import Process

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(10, 3, sparse=True)
        self.net = nn.Linear(2, 3)

    def forward(self, x):
        return self.net(x) + self.embedding(torch.tensor(0).cuda())


def init_processes(rank, size, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.set_device(rank)
    model = nn.parallel.DistributedDataParallel(Model().cuda(), device_ids=[rank], output_device=0)
    x = torch.rand(20, 2).cuda()
    y = model(x)
    y.mean().backward()


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()