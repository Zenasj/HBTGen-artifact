import torch
import torch.distributed as dist


if __name__ == '__main__':
    dist.init_process_group("nccl", init_method='env://')

    # rank = dist.get_rank()
    # local_rank = rank % torch.cuda.device_count()
    # torch.set_default_device(f"cuda:{local_rank}")
    # torch.cuda.set_device(local_rank)

    dist.barrier()

import logging
import torch
import torch.distributed as dist


def print_rank_0(msg, *args, **kwargs):
    rank = dist.get_rank()
    if rank == 0:
        logging.info(msg, *args, **kwargs)


def dist_allreduce():
    print_rank_0("all_reduce:")
    print(torch.get_default_device())

    rank = dist.get_rank()

    dist.barrier()

    tensor = torch.tensor(rank)
    input_tensor = tensor.clone()
    print(input_tensor)
    dist.all_reduce(tensor)

    logging.info(
        f"all reduce, rank: {rank}, before allreduce tensor: {repr(input_tensor)}, after allreduce tensor: {repr(tensor)}")
    dist.barrier()
    print(torch.cuda.current_device())


def main():
    dist.init_process_group("nccl", init_method='env://')

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    torch.set_default_device(f"cuda:{local_rank}")

    print(torch.cuda.current_device())
    # torch.cuda.set_device(local_rank)
    dist_allreduce()


if __name__ == '__main__':
    logging.basicConfig(format=logging.BASIC_FORMAT, level=logging.INFO)
    main()