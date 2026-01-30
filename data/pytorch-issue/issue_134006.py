import torch.distributed as dist
from datetime import timedelta
import os
import time
import torch 

# Enviromnent variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

dist.init_process_group(backend='nccl', timeout=timedelta(seconds=10))

n=4 #Try: 1, 2, 4, 8

# Create subgroups
ranks_subgroups = [[r for r in range(world_size) if r%n==rank%n] for rank in range(n)]
if rank==0:
    print(ranks_subgroups)

group_DDP_torch, subgroups  = dist.new_subgroups_by_enumeration(ranks_subgroups)
device = f'cuda:{local_rank}'

print('Worker '+str(rank)+' enters here.')
input_slice = torch.zeros(1).cuda(device)
output = torch.zeros(world_size//n).cuda(device)

dist.barrier(group=group_DDP_torch)
dist.all_gather_into_tensor(output, input_slice, group=group_DDP_torch)
print('Worker '+str(rank)+' was able to gather.')
dist.barrier(group=group_DDP_torch)

dist.destroy_process_group()

import torch.distributed as dist
from datetime import timedelta
import os
import time
import torch

# Enviromnent variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])

dist.init_process_group(backend='nccl', timeout=timedelta(seconds=10))

n=4 #Try: 1, 2, 4, 8

# Create subgroups
ranks_subgroups = [[r for r in range(world_size) if r%n==rank%n] for rank in range(n)]
if rank==0:
    print(ranks_subgroups)

group_DDP_torch, subgroups  = dist.new_subgroups_by_enumeration(ranks_subgroups)
device = f'cuda:{local_rank}'
torch.cuda.set_device(device)

print('Worker '+str(rank)+' enters here.')
input_slice = torch.zeros(1).cuda(device)
output = torch.zeros(world_size//n).cuda(device)

# Replace the first ever barrier with a single-element all-reduce.
# This is basically what ProcessGroupNCCL::barrier() does but without the guessing.
barrier_tensor = torch.zeros(1).cuda(device)
dist.all_reduce(barrier_tensor, group=group_DDP_torch)

dist.all_gather_into_tensor(output, input_slice, group=group_DDP_torch)
torch.cuda.synchronize()
print('Worker '+str(rank)+' was able to gather.')

# Subsequent barrier should work
dist.barrier(group=group_DDP_torch)

dist.destroy_process_group()