import os
import torch
import torch.distributed as dist

def distributed_init(rank, local_rank, world_size):
    dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=world_size)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.all_reduce(torch.zeros(1).cuda())

def distributed_main(rank, local_rank, world_size):
    distributed_init(rank, local_rank, world_size)
    tmp_ranks = list(range(world_size))
    tmp_group = dist.new_group(tmp_ranks)
    dist.barrier(tmp_group)

if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    print("rank: ", rank, "local_rank: ", local_rank, "world_size: ", world_size)
    distributed_main(rank, local_rank, world_size)