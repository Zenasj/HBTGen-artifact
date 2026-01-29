# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.distributed as dist
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChunkShardingSpec:
    dim: int
    placements: list

    def build_metadata(self, size, tensor_properties):
        # Placeholder for building metadata
        return ShardedTensorMetadata(size, tensor_properties)

    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None) -> Optional["ShardedTensor"]:
        from torch.distributed._shard.sharded_tensor import ShardedTensor, Shard
        from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata, TensorProperties

        tensor_properties = TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )
        current_rank = dist.get_rank(process_group)
        tensor_meta = self.build_metadata(tensor.size(), tensor_properties)
        local_shards = []
        local_tensor = None
        local_metadata = None
        tensors_to_scatter = [None] * dist.get_world_size(process_group)

        sharding_dim_size = tensor.size()[self.dim]
        chunks = len(self.placements)
        split_size = sharding_dim_size // chunks
        scatter_shape = list(tensor.size())
        scatter_shape[self.dim] = split_size

        for shard_meta in tensor_meta.shards_metadata:
            rank, device = _parse_and_validate_remote_device(process_group, shard_meta.placement)
            if current_rank == src_rank:
                narrowed_tensor = tensor.narrow(self.dim, shard_meta.shard_offsets[self.dim], shard_meta.shard_sizes[self.dim]).detach().clone()
                if shard_meta.shard_sizes[self.dim] < split_size:
                    tensor_to_scatter = narrowed_tensor.resize_(scatter_shape)
                else:
                    tensor_to_scatter = narrowed_tensor.contiguous()

                tensors_to_scatter[rank] = tensor_to_scatter

            if current_rank == rank:
                local_tensor = torch.empty(scatter_shape, dtype=tensor.dtype, layout=tensor.layout, device=device)
                local_metadata = shard_meta

        for rank in range(dist.get_world_size(process_group)):
            tensor_to_scatter = tensors_to_scatter[rank]
            if tensor_to_scatter is None:
                tensor_to_scatter = torch.empty(scatter_shape, dtype=tensor.dtype, device="cuda")
                tensors_to_scatter[rank] = tensor_to_scatter
                if current_rank == rank:
                    local_tensor = torch.empty(scatter_shape, dtype=tensor.dtype, layout=tensor.layout, device="cuda")

        assert local_tensor is not None

        src_for_scatter = src_rank
        if process_group is not None and process_group is not dist.group.WORLD:
            src_for_scatter = dist.get_global_rank(process_group, src_for_scatter)

        dist.scatter(local_tensor, scatter_list=tensors_to_scatter if current_rank == src_rank else None, src=src_for_scatter, group=process_group)

        if local_metadata is None:
            return None

        if list(local_tensor.size()) != local_metadata.shard_sizes:
            local_tensor = local_tensor.resize_(local_metadata.shard_sizes).detach()

        local_tensor.requires_grad = tensor.requires_grad

        local_shards.append(Shard(tensor=local_tensor, metadata=local_metadata))

        st = ShardedTensor._init_from_local_shards_and_global_metadata(local_shards, tensor_meta, process_group=process_group)
        st._sharding_spec = self

        return st

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 128, dtype=torch.float32)

def _parse_and_validate_remote_device(process_group, placement):
    # Placeholder for parsing and validating remote device
    return 0, "cuda:0"

def _alloc_tensor(properties, size):
    # Placeholder for allocating tensor
    return torch.empty(size, dtype=properties.dtype, layout=properties.layout, device="cuda")

def _shard_tensor(tensor, sharding_spec):
    # Placeholder for sharding tensor
    return sharding_spec.shard(tensor)

def load_sharded_optimizer_state_dict(state_dict, sharding_spec):
    for key, value in state_dict.items():
        sharded_tensor = _shard_tensor(_alloc_tensor(value.properties, value.size), sharding_spec)
        if sharded_tensor is not None:
            state_dict[key] = sharded_tensor
    return state_dict

