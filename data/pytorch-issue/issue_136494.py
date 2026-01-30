import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import enable_symm_mem_for_group, get_symm_mem_workspace

dist.init_process_group(backend='nccl')

rank = dist.get_rank()
torch.cuda.set_device(rank)

group_name = dist.group.WORLD.group_name
enable_symm_mem_for_group(group_name)

tensor = torch.randn(2, 2)
symm_mem = get_symm_mem_workspace(
    group_name, tensor.numel() * tensor.element_size()
)