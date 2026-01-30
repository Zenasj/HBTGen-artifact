import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def all_reduce_demo(rank, world_size):
    setup(rank, world_size)
    
    torch.cuda.set_device(rank)
    dtype = torch.float32 if rank != 0 else torch.bfloat16
    # dtype = torch.float16 
    tensor = torch.tensor([1e-1], dtype=dtype).cuda()
    print(f"Rank {rank} before all_reduce: {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} after all_reduce: {tensor}")

    cleanup()

if __name__ == "__main__":
    world_size = 3
    mp.spawn(all_reduce_demo, args=(world_size,), nprocs=world_size, join=True)