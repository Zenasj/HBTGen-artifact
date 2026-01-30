import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    ShardingStrategy,
)
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


class Model(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.layer1 = nn.Linear(size // 4, 1, bias=False)
        self.layer2 = nn.Linear(1, size // 4, bias=False)
        self.layer3 = nn.Linear(size // 4, 1, bias=False)
        self.layer4 = nn.Linear(1, size // 4, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_auto_wrap_policy():
    return functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000,  # 自动包装大于100M参数的层
        force_leaf_modules=set(),
        exclude_wrap_modules=set()
    )


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    num_devices = 8
    shard_size = 4

    shard_groups = tuple(
        dist.new_group(backend="nccl", ranks=list(range(group, group + shard_size)))
        for group in range(0, num_devices, shard_size)
    )
    current_shard_group = shard_groups[rank // shard_size]

    replicate_groups = tuple(
        dist.new_group(backend="nccl", ranks=list(range(sz, world_size, shard_size)))
        for sz in range(shard_size)
    )
    current_replicate_group = replicate_groups[rank % shard_size]

    size_in_GB = 4
    model_size = size_in_GB * 1024 * 1024 * 1024 // 4
    if rank % shard_size == 0:
        with torch.device("cuda"):
            model = Model(model_size)
        model_init_fn = None
    else:
        with torch.device("meta"):
            model = Model(model_size)
        model_init_fn = lambda x: x.to_empty(
            device=torch.cuda.current_device(), recurse=False
        )


    model = FSDP(
        model,
        process_group=(
            current_shard_group,
            current_replicate_group
        ),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        param_init_fn=model_init_fn,
        auto_wrap_policy=get_auto_wrap_policy(),
        sync_module_states=True,
    )

    # This is ok, yet it gives deprecated warning
    # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    #     state_dict = model.state_dict()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # This will raise error
    state_dict, optimizer = get_state_dict(model, optimizer)

    if rank == 0:
        from safetensors.torch import save_file
        save_file(state_dict, f"./model.safetensors")

    for pg in shard_groups:
        dist.destroy_process_group(pg)
    for pg in replicate_groups:
        dist.destroy_process_group(pg)
    dist.destroy_process_group()

import torch
import torch.nn as nn
import torch.distributed as dist

import functools
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


class Model(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.layer1 = nn.Linear(size // 4, 1, bias=False)
        self.layer2 = nn.Linear(1, size // 4, bias=False)
        self.layer3 = nn.Linear(size // 4, 1, bias=False)
        self.layer4 = nn.Linear(1, size // 4, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def get_auto_wrap_policy():
    return functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=100_000_000,
        force_leaf_modules=set(),
        exclude_wrap_modules=set(),
    )


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    num_devices = 8
    shard_size = 4

    shard_groups = tuple(
        dist.new_group(backend="nccl", ranks=list(range(group, group + shard_size)))
        for group in range(0, num_devices, shard_size)
    )
    current_shard_group = shard_groups[rank // shard_size]

    replicate_groups = tuple(
        dist.new_group(backend="nccl", ranks=list(range(sz, world_size, shard_size)))
        for sz in range(shard_size)
    )
    current_replicate_group = replicate_groups[rank % shard_size]

    size_in_GB = 4
    model_size = size_in_GB * 1024 * 1024 * 1024 // 4
    if rank % shard_size == 0:
        with torch.device("cuda"):
            model = Model(model_size)
        model_init_fn = None
    else:
        with torch.device("meta"):
            model = Model(model_size)
        model_init_fn = lambda x: x.to_empty(
            device=torch.cuda.current_device(), recurse=False
        )

    model = FSDP(
        model,
        process_group=(current_shard_group, current_replicate_group),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        param_init_fn=model_init_fn,
        auto_wrap_policy=get_auto_wrap_policy(),
        sync_module_states=True,
    )

    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # This gives me a warning
    state_dict = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))

    for pg in shard_groups:
        dist.destroy_process_group(pg)
    for pg in replicate_groups:
        dist.destroy_process_group(pg)
    dist.destroy_process_group()