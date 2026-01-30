mean, invstd = torch.batch_norm_gather_stats(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            int(input.numel() / input.size(1))
        )

"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP


def run(rank, config):
    m = nn.SyncBatchNorm(2, momentum=0.99).to(config["device_ids"][rank])
    ddp_model = DDP(m)
    for i in range(100):
        I = torch.ones(2, 2, config["sizes"][rank]) * config["values"][rank]
        O = ddp_model(I)
    print(rank, ddp_model.module.running_mean.sum(), ddp_model.module.running_var.sum())


def init_processes(rank, size, fn, config, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.manual_seed(1234)
    fn(rank, config)


if __name__ == "__main__":
    
    config = {
        "device_ids": [0, 0, 0],
        "sizes": [10, 10, 10],
        # "sizes": [10, 50, 100],
        "values": [1, 0.1, 0.01],
    }
    processes = []
    world_size = 3
    for rank in range(world_size):
        p = Process(target=init_processes, args=(rank, world_size, run, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()