import torch
import os
import torch.distributed as dist

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

if __name__ == "__main__":
    dist.init_process_group("gloo")
    t = torch.zeros(2)
    t_list = [torch.zeros(2) for _ in range(1)]
    print(f"{t_list=}")
    dist.all_gather(t_list, t)