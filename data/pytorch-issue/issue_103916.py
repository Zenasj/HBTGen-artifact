# python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 this_script.py

import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    a = torch.tensor([[0, 2.], [3, 0]]).to(rank)
    a = a.to_sparse()
    print(f"rank {rank} - a: {a}")
    dist.all_reduce(a)

if __name__ == "__main__":
    main()