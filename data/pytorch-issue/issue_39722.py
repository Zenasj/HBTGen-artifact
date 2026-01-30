"""pytorch test code"""
import os
import argparse
import time
import torch

world_size = 2

def build_model(gpu):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.cuda.set_device(gpu)
    print(f"rank = {gpu}")

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=gpu)

    print(f"BARRIER UP -> GPU:{gpu}")
    torch.distributed.barrier()
    print(f"BARRIER DOWN -> GPU:{gpu}")

def main():
    torch.multiprocessing.spawn(build_model, nprocs=world_size, join=True)


if __name__ == '__main__':
    main()