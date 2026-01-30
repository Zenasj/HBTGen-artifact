import torch

from torch.distributed import checkpoint as dist_cp
dist_cp.load(
  state_dict=state_dict,
  storage_reader=storage_reader,
  planner=load_planner,
)