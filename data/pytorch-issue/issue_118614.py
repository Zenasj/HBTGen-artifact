import torch

from torch.distributed.device_mesh import  DeviceMesh
sub_mesh_list = [0, 2]
sub_mesh = DeviceMesh("cuda", sub_mesh_list)

print(f"rank:{self.rank}, {sub_mesh.get_group()=}")