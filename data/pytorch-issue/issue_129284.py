import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl", device_id =torch.device(f"cuda:{torch.distributed.get_node_local_rank(0)}"))
dist.new_group([0, 1], group_desc="lows")
dist.new_group([2, 3], group_desc="highs")
print("Here")
torch.distributed.destroy_process_group()