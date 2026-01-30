import torch

def worker(rank, world_size):
    # Above code omitted...
    if rank == 0:
        dst_worker_rank = (rank + 1) % world_size
        dst_worker_name = f"worker{dst_worker_rank}"
        t1, t2 = random_tensor(), random_tensor()
        # Send and wait RPC completion under profiling scope.
        with profiler.profile() as prof:
            fut1 = rpc.rpc_async(dst_worker_name, torch.add, args=(t1, t2))
            fut2 = rpc.rpc_async(dst_worker_name, torch.mul, args=(t1, t2))
            # RPCs must be awaited within profiling scope.
            fut1.wait()
            fut2.wait()

        print(prof.key_averages().table())

def worker(rank, world_size):
    # Above code omitted...
    if rank == 0:
        dst_worker_rank = (rank + 1) % world_size
        dst_worker_name = f"worker{dst_worker_rank}"
        t1, t2 = random_tensor(), random_tensor()
        # Send and wait RPC completion under profiling scope.
        with profiler.profile() as prof:
            fut1 = rpc.rpc_async(dst_worker_name, torch.add, args=(t1, t2))
            fut2 = rpc.rpc_async(dst_worker_name, torch.mul, args=(t1, t2))
            # RPCs must be awaited within profiling scope.
            fut1.wait()
            fut2.wait()
            # print vals
            print(fut1.value())
            print(fut2.value())

        print(prof.key_averages().table())

def _get_should_profile():
    # Legacy profiler should be enabled. RPC profiling is not supported with
    # Kineto profiler.
    ActiveProfilerType = torch._C._profiler.ActiveProfilerType

    # print vars
    print(torch.autograd._profiler_enabled())         # True
    print(torch._C._autograd._profiler_type())        # ActiveProfilerType.KINETO
    print(ActiveProfilerType.LEGACY)                  # ActiveProfilerType.LEGACY

    return (
        torch.autograd._profiler_enabled() and
        torch._C._autograd._profiler_type() == ActiveProfilerType.LEGACY  # type: ignore[attr-defined]
    )