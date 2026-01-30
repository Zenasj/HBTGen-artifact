3
# NCCL_DEBUG=INFO torchrun --nproc-per-node 4 test.py
import os
import torch.distributed as dist

dist.init_process_group(backend='nccl')

rank = int(os.environ["RANK"])

if rank == 0 or rank == 3:
    grp_ranks = [0, 3]
else:
    assert rank == 1 or rank == 2
    grp_ranks = [1, 2]

grp = dist.new_group(ranks=grp_ranks)

print(f"{dist.get_rank()} before barrier: {dist.get_process_group_ranks(grp)}")
dist.barrier(group=grp)
print(f"{dist.get_rank()} done")

3
# NCCL_DEBUG=INFO torchrun --nproc-per-node 4 test.py
import os
from time import sleep
import torch.distributed as dist

dist.init_process_group(backend='nccl')

rank = int(os.environ["RANK"])

if rank == 0 or rank == 3:
    grp_ranks = [0, 3]
    sleep(1)  # let 1 and 2 reach barrier first
else:
    assert rank == 1 or rank == 2
    grp_ranks = [1, 2]

grp = dist.new_group(ranks=grp_ranks)

print(f"{dist.get_rank()} before barrier: {dist.get_process_group_ranks(grp)}")
dist.barrier(group=grp)
print(f"{dist.get_rank()} done")