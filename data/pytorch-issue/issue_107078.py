import torch
import torch.distributed as dist

import argparse
import os

def main(rank):
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(rank)

    g = dist.new_group(backend='gloo')
    print(dist.get_backend(g), type(g), isinstance(g, dist.ProcessGroupGloo), isinstance(g, dist.ProcessGroupNCCL))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', '0'))
    args = parser.parse_args()

    # print(args)
    rank = int(args.local_rank)

    main(rank)