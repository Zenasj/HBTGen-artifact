import torch

def init_dist(draft_rank, target_rank) -> Optional[int]:
    global_rank = _get_global_rank()
    world_size = _get_world_size()
    torch.cuda.set_device(global_rank)
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)
    draft_group = dist.new_group(draft_rank, backend="nccl")
    target_group = dist.new_group(target_rank, backend="nccl")
    return global_rank, draft_group, target_group