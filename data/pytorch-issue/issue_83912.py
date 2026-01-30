import torch
import torch.distributed as dist
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

def main():
    dist.init_process_group("gloo")
    print("finished creating process group")
    t = torch.tensor([1, 2, 3])
    dist.send(t, 0)

if __name__ == "__main__":
    main()