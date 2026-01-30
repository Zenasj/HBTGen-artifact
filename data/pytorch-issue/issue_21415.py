import torch.distributed.distributed_c10d as c10d
list(c10d._pg_group_ranks[pg].values())