import torch
import torch.nn as nn

def init_orthogonal(tensor, gain):
    if isinstance(tensor, DTensor):
        full_tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        if dist.get_rank() == 0:
            torch.nn.init.orthogonal_(full_tensor, gain=gain)
        dist.broadcast(
            full_tensor,
            0,
            group=dist.group.WORLD,
        )
        tensor.copy_(distribute_tensor(full_tensor, device_mesh=tensor.device_mesh, placements=tensor.placements))
    else:
        torch.nn.init.orthogonal_(tensor, gain=gain)