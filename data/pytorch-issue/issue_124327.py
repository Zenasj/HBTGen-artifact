import torch

device = f'cuda:{rank}'
torch.cuda.set_device(device)
device_mesh = init_device_mesh(
    "cuda", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
)

#   File "/data/users/whc/pytorch/torch/distributed/_composable/fsdp/_fsdp_init.py",
#   line 69, in _get_device_from_mesh
#              return torch.device(mesh.device_type, device_handle.current_device())
#   AttributeError: 'NoneType' object has no attribute 'current_device'
device_mesh = init_device_mesh(
    "cuda", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names
)