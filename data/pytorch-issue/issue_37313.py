import torch.nn as nn

import logging
import torch
import torchvision
import comm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel


def main():

    model = torchvision.models.resnet50(pretrained=False) # skip model download
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(comm.get_local_rank())
    model = DistributedDataParallel(
        model,
        device_ids=[comm.get_local_rank()],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
           
    profiler = torch.autograd.profiler.profile(True, True, True)
    comm.synchronize()

    for i in range(100000):
        print(i)
        dummpy_input = torch.zeros(4,3,400,400)
        if i > 10 and i < 20:
            with torch.autograd.profiler.profile(True, True, True) as prof:
                out = model(dummpy_input)
        else:
            out = model(dummpy_input)


    print("rank {}: finish".format(comm.get_local_rank()))

def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

def _distributed_worker(
    local_rank, main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url, world_size=world_size, rank=global_rank
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    main_func(*args)

if __name__ == "__main__":
    port = _find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"

    mp.spawn(
        _distributed_worker,
        nprocs=2,
        args=(main, 2, 2, 0, dist_url, ()),
        daemon=False,
    )

import torch.distributed as dist


_LOCAL_PROCESS_GROUP = None


# some codes for maskrcnn
def get_local_rank() -> int:
    """The rank of the current process within the local machine.

    Returns:
        int: The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)

def synchronize():
    """Synchronization Function.

    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()