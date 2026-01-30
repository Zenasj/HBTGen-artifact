import torch

mesh = torch.arange(0, world_size).view(mp_size, dp_size).transpose(0, 1)
device_mesh = DeviceMesh(
                "cuda",
mesh.contiguous(),
mesh_dim_names=("dp", "mp")
)

mesh = torch.arange(0, world_size).view(mp_size, dp_size).transpose(0, 1)
device_mesh = DeviceMesh(
                "cuda",
mesh.contiguous(),
mesh_dim_names=("dp", "mp")
)