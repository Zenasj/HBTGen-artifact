import torch
import random

world_mesh = init_device_mesh(
    device_type="cuda",
    mesh_shape=(2, 2, 2),
    mesh_dim_names=("pp", "dp", "tp"),
)
pp_mesh = world_mesh["pp"]
pp_rank = pp_mesh.get_local_rank()
spmd_mesh = world_mesh["dp", "tp"]._flatten("spmd")  # this flattening is only needed if you need to call collective over this mesh
torch.distributed.tensor._random.manual_seed(123+pp_rank, spmd_mesh)