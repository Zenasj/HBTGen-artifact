"""
Environment:
mamba create -y -n tmp \
    -c pytorch -c nvidia -c conda-forge \
    python tabulate pytorch=2.1.0 pytorch-cuda=12.1
mamba activate tmp

One GPU:
    python checkpoint.py

Two GPUs:
    WORLD_SIZE=2 RANK=0 python checkpoint.py
    WORLD_SIZE=2 RANK=1 python checkpoint.py
"""

import contextlib
import os

import tabulate
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType


def main():
    # Dist
    device = init_dist()
    dtype = torch.bfloat16
    torch.manual_seed(42 + dist.get_rank())

    # Model
    model = get_model()
    x = torch.tensor([1.0, 2.0])

    # State dict before FSDP
    state_dict = model.state_dict()
    print("State dict before FSDP", to_table(state_dict), sep="\n", end="\n\n")

    # Forward before FSDP
    # (will be different in each rank)
    y = model(x)
    print(f"Forward before FSDP: {y.item()}", end="\n\n")

    # FSDP
    model = wrap_fsdp(model, device, dtype)

    # Forward after FSDP
    # (must match across ranks)
    y = model(x)
    print(f"Forward after FSDP: {y.item()}", end="\n\n")

    # Save per-rank checkpoint
    print("Save per-rank checkpoint")
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        state_dict = model.state_dict()
        print(to_table(state_dict), end="\n\n")
        torch.save(state_dict, f"/tmp/rank_{dist.get_rank()}.pth")

    # Re-initialize model and wrap with FSDP
    model = get_model()
    model = wrap_fsdp(model, device, dtype)

    # Forward after re-initialization
    # (must match across ranks but can different from before)
    y = model(x)  # <--
    print(f"Forward after re-initialization: {y.item()}", end="\n\n")  # <--
    # y.sum().backward()  # <--

    # Load per-rank checkpoint
    print("Load per-rank checkpoint")
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        state_dict = torch.load(f"/tmp/rank_{dist.get_rank()}.pth")
        print(to_table(state_dict), end="\n\n")
        res = model.load_state_dict(state_dict, strict=False)
        print("Load result:", res, end="\n\n")

    # Forward after loading
    # (must match across ranks and be the same as before saving)
    y = model(x)
    print(f"Forward after loading: {y.item()}", end="\n\n")


def init_dist():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "54321"))
    device = torch.device("cuda", rank % torch.cuda.device_count())
    init_method = f"tcp://{master_addr}:{master_port}"
    print(f"Init dist: {init_method}, rank {rank}/{world_size}")
    dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=world_size)
    dist.all_reduce(torch.tensor(1.0, device=device))
    torch.cuda.set_device(device)
    return device


def get_model():
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1, bias=False),
    )


def wrap_fsdp(model, device, dtype):
    return FSDP(
        model,
        device_id=device,
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
        sync_module_states=True,
    )


def to_table(state_dict):
    def sum_(p):
        with contextlib.suppress(RuntimeError):
            return p.sum().item()

    tab = [
        {
            "name": n,
            "dtype": p.dtype,
            "device": p.device,
            "shape": list(p.shape),
            "sum": sum_(p),
        }
        for n, p in state_dict.items()
    ]
    return tabulate.tabulate(tab, headers="keys")


if __name__ == "__main__":
    main()

def reshard_fsdp(model):
      for m in FullyShardedDataParallel.fsdp_modules(model):
          if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
              torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)