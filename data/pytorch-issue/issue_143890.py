import functools
import os

import torch


@functools.cache
def world_group() -> torch.distributed.ProcessGroup:
    """Get NCCL process group, initializing if needed"""
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    group = torch.distributed.init_process_group(
        "nccl",
        init_method="file:///tmp/rdzv",
        world_size=world_size,
        rank=rank,
    )
    return group


def main() -> None:

    # Parallel config
    group = world_group()
    group_size = torch.distributed.get_world_size(group)
    rank = torch.distributed.get_rank(group)

    # Buffer for communication
    all_gather_size = 2 ** 30
    all_gather_buffer = torch.zeros(
        all_gather_size * group_size,
        dtype=torch.uint8,
        device="cuda",
    )
    all_gather_buffer_local = all_gather_buffer[rank * all_gather_size : (rank+1) * all_gather_size]

    # Buffer for compute
    compute_size = 2 ** 30
    compute_buffer = torch.ones(compute_size, dtype=torch.float32, device="cuda")

    # Overlap communication and computation
    default_stream = torch.cuda.default_stream()
    comm_stream = torch.cuda.Stream()
    for _ in range(10):
        comm_stream.wait_stream(default_stream)
        default_stream.wait_stream(comm_stream)
        with torch.cuda.stream(comm_stream):
            torch.distributed.all_gather_into_tensor(
                all_gather_buffer,
                all_gather_buffer_local,
                group=group,
            )
        torch.cos(compute_buffer)
    torch.cuda.synchronize()

    # Print status
    print(f"{rank=}: finished...")

if __name__ == "__main__":
    main()