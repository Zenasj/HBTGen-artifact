import torch.nn as nn

import os

import torch
import torch.distributed as dist
import torch.distributed._state_dict_utils
import torch.multiprocessing as mp


def run(rank, size):
    """Distributed function to be implemented later."""
    torch.distributed._state_dict_utils._broadcast_tensors(
        {"key": torch.zeros(1)}, local_state_dict={}, device=torch.device("cpu"), keys=["key"]
    )


def init_process(rank, size, fn):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def _broadcast_tensors(
    full_state_dict: dict[str, Any],
    local_state_dict: dict[str, Any],
    keys: list[str],
    device: torch.device,
    pg: torch.distributed.ProcessGroup | None = None,
) -> None:
    tensors = []
    for key in keys:
        if torch.distributed.get_rank() == 0:
            full_state = full_state_dict[key]
            assert isinstance(full_state, torch.Tensor)
            full_tensor = full_state.detach().to(device)
        else:
            tensor_info = full_state_dict[key]
            full_tensor = torch.empty(
                size=tensor_info.size,
                device=device,
                dtype=tensor_info.dtype,
            )

        tensors.append(full_tensor)
        local_state = local_state_dict.get(key)
        if local_state is None:
            continue
        if isinstance(local_state, torch.distributed._tensor.DTensor):  # pyright: ignore[reportPrivateImportUsage]
            local_state_dict[key] = (local_state, full_tensor)
        else:
            local_state_dict[key] = full_tensor

    if pg is None:
        pg = torch.distributed.distributed_c10d._get_default_group()

    tensors = [tensor.to(pg._device_types[0]) for tensor in tensors] # cast to the process group device
    if len(tensors) > 1:
        torch.distributed._broadcast_coalesced(pg, tensors, 500, 0)  # pyright: ignore[reportPrivateImportUsage]
    else:
        torch.distributed.broadcast(tensors[0], src=0, group=pg)
    tensors = [tensor.to(device) for tensor in tensors] # cast back to the original device

    # Because the way the code prior to the broadcast operates by reference, we need to redefine these elements prior to distribution
    if isinstance(local_state_dict[keys[0]][0], torch.distributed._tensor.DTensor):  # pyright: ignore[reportPrivateImportUsage]
        local_state_dict[keys[0]] = (local_state_dict[keys[0]][0], tensors[0])
    else:
        local_state_dict[keys[0]] = tensors[0]

    torch.distributed._state_dict_utils._distribute_tensors(local_state_dict, keys, device, pg)

"""Minimum reproducible example for https://github.com/pytorch/pytorch/issues/138842.

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=6006 mre.py
"""
from typing import Any, Callable

import torch
import torch.distributed.checkpoint.state_dict
from torch import nn


def manage_process_group(func: Callable[..., Any]) -> Callable[..., Any]:
    """Manage the creation and destruction of the distributed process group for the wrapped function."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        torch.distributed.init_process_group(world_size=torch.cuda.device_count())
        torch.cuda.set_device(torch.distributed.get_rank())
        try:
            return func(*args, **kwargs)
        finally:
            torch.distributed.destroy_process_group()

    return wrapped


@manage_process_group
def main() -> None:
    model = nn.Linear(2, 2)
    checkpoint = "model.pt"
    if torch.distributed.get_rank() == 0:
        torch.save(model.state_dict(), checkpoint)

    state_dict = (
        torch.load(checkpoint)
        if torch.distributed.get_rank() == 0
        else {}
    )
    options = torch.distributed.checkpoint.state_dict.StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    torch.distributed.checkpoint.state_dict.set_model_state_dict(model, state_dict, options=options)


if __name__ == "__main__":
    main()