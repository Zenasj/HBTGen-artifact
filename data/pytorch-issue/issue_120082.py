import torch
import torch._dynamo.testing
import torch.distributed as dist


def tensor_to_amax(x: torch.Tensor):
    amax = torch.max(torch.abs(x))
    dist.all_reduce(amax, op=dist.ReduceOp.MAX)
    return amax


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
    x = torch.ones((1,), device="cuda") * (rank + 1)
    amax = torch.compile(tensor_to_amax, backend=cnt)(x)
    assert amax.item() == dist.get_world_size(), f"[Rank {rank}] amax: {amax.item()}"
    print(f"[Rank {rank}] frame count: {cnt.frame_count}")

import torch
import torch._dynamo.testing
import torch.distributed as dist


def tensor_to_amax(x: torch.Tensor):
    amax = torch.max(torch.abs(x))
    dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=dist.distributed_c10d._get_default_group())
    return amax


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
    x = torch.ones((1,), device="cuda") * (rank + 1)
    amax = torch.compile(tensor_to_amax, backend=cnt)(x)
    assert amax.item() == dist.get_world_size(), f"[Rank {rank}] amax: {amax.item()}"
    print(f"[Rank {rank}] frame count: {cnt.frame_count}")

NotImplementedError: UserDefinedObjectVariable(RedOpType)