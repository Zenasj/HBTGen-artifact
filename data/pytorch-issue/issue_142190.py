import os
import torch
import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor, Shard, Replicate


torch.manual_seed(0)

def main(rank, world_size):
    device = torch.device("cuda:%d" % rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device,
    )
    mesh = dist.init_device_mesh("cuda", (world_size,))

    dim = 128
   
    x = torch.randn(8, dim, device=device)
    A = torch.randn(dim, dim, device=device)
    y = torch.matmul(x, A)

    # DTensor test
    dx = distribute_tensor(x, mesh, [Replicate()])
    dA = distribute_tensor(A, mesh, [Shard(0)])
    with torch.inference_mode():
        dy = torch.ops.aten.matmul.default(dx, dA)

    torch.testing.assert_close(y, dy.full_tensor())

    dist.destroy_process_group()
    print("clean exit")

if __name__ == "__main__":
    main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))