import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import os
import tempfile

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(
                backend="gloo",
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )

    backend_opts = rpc.ProcessGroupRpcBackendOptions(init_method="tcp://localhost:29500")
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=backend_opts,
    )

    print("Done with init")

    rpc.shutdown()


if __name__ == '__main__':
    ws = 2
    mp.spawn(run, args=(ws,), nprocs=2)