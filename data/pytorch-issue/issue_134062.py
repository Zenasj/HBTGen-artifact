import torch
import os
import torch.distributed.distributed_c10d as c10d

def repro(rank, world_size):
    os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
    device = torch.device("cuda:%d" % rank)
    c10d.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )
    
    x = torch.ones((10,), dtype=torch.float32, device=device)
    c10d.all_reduce(x)
    print(f"After first allreduce: {x=}")
    c10d.all_reduce(x)
    print(f"After second allreduce: {x=}")
    torch.cuda.synchronize()
    c10d.destroy_process_group()
    print("clean exit")

if __name__ == "__main__":
    repro(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]))

torch.cuda.set_device(device)