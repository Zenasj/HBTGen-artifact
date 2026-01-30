import torch.distributed as dist
dist.init_process_group(backend='mpi', rank=0, world_size=1)