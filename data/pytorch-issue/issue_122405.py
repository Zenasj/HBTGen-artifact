import torch

@torch.compile()
def get_traj_idx(lengths: torch.Tensor, num_slices: int) -> torch.Tensor:
    return torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)

lengths = torch.zeros(10, dtype=torch.long)
get_traj_idx(lengths, num_slices=4)
lengths = torch.zeros(11, dtype=torch.long)
get_traj_idx(lengths, num_slices=4)

import torch

@torch.compile()
def get_traj_idx(lengths: torch.Tensor, num_slices: int) -> torch.Tensor:
    # return torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)
    return (torch.rand((num_slices,), device=lengths.device) * lengths.shape[0]).floor()

lengths = torch.zeros(10, dtype=torch.long)
get_traj_idx(lengths, num_slices=4)
lengths = torch.zeros(11, dtype=torch.long)
get_traj_idx(lengths, num_slices=4)