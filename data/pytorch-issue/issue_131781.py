import torch
import torch.distributed as dist
import os

def run(local_rank, global_rank):
    print(f"Running basic all-reduce on global rank {global_rank}.")
    torch.cuda.set_device(local_rank)
    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Local rank {local_rank} has data {tensor[0]} after all-reduce.")

def main():
    os.environ['MASTER_ADDR'] = 'g105'
    os.environ['MASTER_PORT'] = '29500'
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    print('Preparing to init...')
    dist.init_process_group('nccl')
    print('Initted')
    run(local_rank, global_rank)

if __name__ == '__main__':
    main()