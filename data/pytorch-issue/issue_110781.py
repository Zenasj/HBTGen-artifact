import torch

tp_size = torch.cuda.device_count()
global_world_size = dist.get_world_size()
assert global_world_size % tp_size == 0  # `.view()` would handle this assertion before
mesh_shape = (global_world_size // tp_size, tp_size)