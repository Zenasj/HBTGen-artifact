import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '17777'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def subprocess_fn(rank):
    setup(rank, 4)
    test_tensor = torch.Tensor([rank]).cuda()
    print(f'Before broadcast data on rank {rank} is {test_tensor}.')
    incorrect_group = dist.new_group([2, 3])
    if rank < 2:
        dist.broadcast(test_tensor, 1, group=incorrect_group)
    print(f'After broadcast data on rank {rank} is {test_tensor}.')


if __name__ == '__main__':
    mp.spawn(subprocess_fn, args=(), nprocs=4)