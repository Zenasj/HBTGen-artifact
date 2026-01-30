import torch

torch.ops.c10d_functional.get_default_process_group()

def _get_default_process_group() -> dist.ProcessGroup:
    return dist.distributed_c10d._get_default_group()

"get_default_process_group() -> __torch__.torch.classes.c10d.ProcessGroup"