py
import torch
import argparse
from torch import distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)

args = parser.parse_args()

torch.distributed.init_process_group("nccl")

local_rank = args.local_rank

device = torch.device(local_rank)

if local_rank == 0:
    element = False
else:
    element = True


def broadcast_scalar(scalar, src=0, device="cpu"):
    scalar_tensor = torch.tensor(scalar).to(device)
    with torch.no_grad():
        scalar_tensor = dist.broadcast(scalar_tensor, src)
    return scalar_tensor.item()


broadcast_scalar(element, src=0, device=device)