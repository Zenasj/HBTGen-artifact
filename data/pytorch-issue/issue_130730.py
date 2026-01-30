import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size):
    rank = rank
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    a = torch.randn(100,300,device='cuda')
    dist.all_reduce(a)
    with torch.profiler.profile() as p:
        dist.all_reduce(a)
        dist.barrier()
    if rank == 0:
        p.export_chrome_trace(f"profile_all_reduce_{rank}.json")
        print(p.key_averages())
    cleanup()
def main():
    world_size = 2
    mp.spawn(demo_basic, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()