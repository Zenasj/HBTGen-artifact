# torch_dist_unit_tests.py

import torch.distributed as dist
import torch

dist.init_process_group(backend='nccl')

process_group = torch.distributed.group.WORLD


if process_group.rank() == 0:   
    tnsr = torch.arange(4, dtype=torch.float32, device=torch.device("cuda:0"))
    tnsr2 = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=torch.device("cuda:0"))
    tnsr3 = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=torch.device("cuda:0"))
    # tnsr4 = torch.tensor([0, 1, 2, 3], dtype=torch.float16, device=torch.device("cuda:0"))
else:
    tnsr = torch.empty(4, device=torch.device("cuda:1"))
    tnsr2 = torch.empty(4, device=torch.device("cuda:1"))
    tnsr3 = torch.empty(4, device=torch.device("cuda:1"))
    # tnsr4 = torch.empty(4, device=torch.device("cuda:1"))

dist.broadcast(tnsr, src=0, group=process_group)
dist.broadcast(tnsr2, src=0, group=process_group)
dist.broadcast(tnsr3, src=0, group=process_group)
# dist.broadcast(tnsr4, src=0, group=process_group)

if process_group.rank() == 1:
    print(f"rank: {process_group.rank()}, {tnsr}")
    print(f"rank: {process_group.rank()}, {tnsr2}")
    print(f"rank: {process_group.rank()}, {tnsr3}")
    # print(f"rank: {process_group.rank()}, {tnsr4}")