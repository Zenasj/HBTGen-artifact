import torch
import torch.multiprocessing as mp
import torch.distributed as c10d

import tempfile

def opts(threads=2):
    opts = c10d.ProcessGroupGloo.Options()
    opts.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
    opts.timeout = 5.0
    opts.threads = threads
    return opts

def reduce_process_gloo(rank, filename, world_size):
    store = c10d.FileStore(filename, world_size)
    pg = c10d.ProcessGroupGloo(store, rank, world_size, opts())
    x = torch.ones(2, 2).to(rank)
    pg.reduce(x, root=0, op=c10d.ReduceOp.SUM).wait()
    print ("gloo rank ", rank, ": x = ", x)

def reduce_process_nccl(rank, filename, world_size):
    store = c10d.FileStore(filename, world_size)
    pg = c10d.ProcessGroupNCCL(store, rank, world_size)
    x = torch.ones(2, 2).to(rank)
    pg.reduce(x, root=0, op=c10d.ReduceOp.SUM).wait()
    print ("nccl rank ", rank, ": x = ", x)

if __name__ == '__main__':
    with tempfile.NamedTemporaryFile(delete=False) as file:
        world_size = 2
        mp.spawn(reduce_process_gloo,
                 args=(file.name, world_size),
                 nprocs=world_size,
                 join=True)

    with tempfile.NamedTemporaryFile(delete=False) as file:
        world_size = 2
        mp.spawn(reduce_process_nccl,
                 args=(file.name, world_size),
                 nprocs=world_size,
                 join=True)