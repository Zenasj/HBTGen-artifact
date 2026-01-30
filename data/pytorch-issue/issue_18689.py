import torch.distributed as dist
import torch as t

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--rank",
            type=int,
            help="the rank of proc"
            )
    args = parser.parse_args()
    dist.init_process_group("nccl", rank=args.rank, world_size=2)
    tmp = [t.randn(5).cuda()] * 2
    tensor = t.ones(5).cuda()
    dist.all_gather(tmp, tensor)
    print(tmp)

if __name__ == "__main__":
    main()