# python -m torch.distributed.run --nproc_per_node=2 <script>.py
import torch
import torch.distributed as dist
import datetime

if __name__ == "__main__":
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=5))
    if dist.get_rank() == 0:
        dist.broadcast(torch.tensor([1, 2, 3]).cuda(), 0)
    print("Rank {} done".format(dist.get_rank()))