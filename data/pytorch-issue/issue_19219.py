import torch
import torch.distributed as dist

def worker(rank):
    for itr in range(1000):
        x = torch.randn(int(25 * 1024 * 1024), device='cuda')  # 25 MiB
        dist.broadcast(x, src=1, async_op=False)
        del x

def main(rank, init_method, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method, rank=rank, world_size=world_size)
    worker(rank)

if __name__ == '__main__':
    init_method = 'tcp://127.0.0.1:23123'
    world_size = 2
    torch.multiprocessing.spawn(main, (init_method, world_size), nprocs=world_size)