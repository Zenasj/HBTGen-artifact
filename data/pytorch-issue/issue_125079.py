import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    device = torch.device(f"cuda:{rank}")
    if rank == 0:
        tens = torch.ones([33, 4096], dtype=torch.bfloat16, device=device)
        tens2 = 2 * torch.ones([33, 4096], dtype=torch.bfloat16, device=device)
        dist.send(tens2, dst=1, tag=0)
        dist.send(tens, dst=1, tag=1)
    else:
        tens = torch.empty([33, 4096], dtype=torch.bfloat16, device=device)
        tens2 = torch.empty([33, 4096], dtype=torch.bfloat16, device=device)
        dist.recv(tens, src=0, tag=1)
        dist.recv(tens2, src=0, tag=0)
        print (f'{tens}, {tens2}')

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()