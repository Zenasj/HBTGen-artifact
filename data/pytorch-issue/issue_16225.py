#debug_dist.py
import argparse
import os

import torch
import torch.nn as nn

def synchronize():
    """
    Helper function to synchronize between multiple processes when
    using distributed training
    """
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return
    torch.distributed.barrier()

def test_inf(caltime):
    rank = torch.distributed.get_rank()
    cpu_device = torch.device("cpu")
    res_list = []
    model = nn.Conv3d(10, 1024, 3).to('cuda')
    model.eval()
    for i in range(caltime):
        input = torch.rand(3, 10, 27, 27, 27, device='cuda')
        with torch.no_grad():
            output = model(input).mean()
            output = output.to(cpu_device)
            res_list.append(output)
        if i%1000==0 and rank==0:
            print(i)
    return res_list

def main():
    parser = argparse.ArgumentParser(description="Debug for distributed inference.")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    rank = torch.distributed.get_rank()
    assert rank == args.local_rank, "damn!"

    caltime = 1000

    res = test_inf(caltime)

    if rank == 0:
        time.sleep(330)

    synchronize()

if __name__ == "__main__":
    main()