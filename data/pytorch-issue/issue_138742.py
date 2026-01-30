import torch
from torch.distributed._tensor import init_device_mesh, Shard, distribute_tensor, Replicate

if __name__ == "__main__":
    mesh = init_device_mesh("cpu", (4,1))
    left = torch.randn(256, 128)
    right = torch.randn(128,256)
    
    left_d = distribute_tensor(left, mesh, [Shard(dim=0), Replicate()])
    right_d = distribute_tensor(right, mesh, [Replicate(), Replicate()])

    res = left_d @ right_d
    print(res)