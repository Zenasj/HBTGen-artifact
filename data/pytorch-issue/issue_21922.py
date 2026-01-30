import os
import socket
import torch
import torch.distributed as dist


def run(rank, size):
    t = torch.rand(1).cuda()
    gather_t = [torch.ones_like(t) for _ in range(size)]
    dist.all_gather(gather_t, t)

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])

    torch.cuda.set_device(1)
    init_processes(world_rank, world_size, run, backend='mpi')