import os
import subprocess
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
    StateDictType
)

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy
)

# import Spur
# from Spur.fsdp import SpurFSDP


def get_master_addr():
    _node_list = os.environ['SLURM_JOB_NODELIST']
    _out = subprocess.run(['scontrol', 'show', 'hostnames', _node_list], capture_output=True, text=True)
    return _out.stdout.split('\n')[0]

def slurm_init():
    master_addr = get_master_addr()
    print(f'SLURM_SRUN_COMM_HOST: {master_addr}')
    master_port = int(os.environ.get("MASTER_PORT", "12340"))
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    dist_url = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=world_size,
        rank=world_rank,
    )

    torch.cuda.set_device(local_rank)


def main():
    slurm_init()
    rank = dist.get_rank()
    linear1 = nn.Linear(8, 32, bias=False)
    linear2 = nn.Linear(32, 32, bias=False)
    model = nn.Sequential(
        linear1,
        nn.ReLU(),
        linear2
    ).cuda()
    if rank == 0:
        torch.save(model.state_dict(), 'model.pth')
    dist.barrier()
    model.load_state_dict(torch.load('model.pth'))

    auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=0)
    mp = MixedPrecision(param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
    fsdp = FSDP(model,
         sharding_strategy=ShardingStrategy.FULL_SHARD, 
         auto_wrap_policy=auto_wrap_policy,
         mixed_precision=mp,
         ignored_modules=[linear2])
    
    x = torch.randn([1, 8]).cuda()
    y = torch.randn([1, 32]).cuda()

    out = fsdp(x)
    loss = F.l1_loss(out, y)
    loss.backward()
    torch.cuda.synchronize()

    grad = linear2.weight.grad.cpu()
    print(f"grad.shape: {grad.shape}")
    torch.save(grad, f'grad.{rank}.pth')
    
    dist.barrier()
    
    if rank == 0:
        grad0 = torch.load('grad.0.pth')
        grad1 = torch.load('grad.1.pth')
        abs_err = torch.abs(grad0 - grad1).max()
        print(f"abs_err: {abs_err}")

    dist.barrier()

if __name__ == '__main__':
    main()