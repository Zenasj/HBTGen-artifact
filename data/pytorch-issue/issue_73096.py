import torch

# Have 2 processes/threads running this function

def rank_build_sharded_tensor(rank: int) -> None:
    shard_tensor = torch.randint(
        low=0,
        high=10,
        size=(1, 4),
    )
    sharding_dim = 0
    sharding_spec = ChunkShardingSpec(
        dim=sharding_dim,
        placements=[
            "rank:0/cpu",
            "rank:1/cpu",
        ],
    )
    shard_offsets = [rank, 0]
    shard_sizes = list(shard_tensor.size())
    shard_metadata = ShardMetadata(
        shard_offsets=shard_offsets,
        shard_sizes=shard_sizes,
        placement=sharding_spec.placements[rank],
    )
    local_shards = [
        sharded_tensor.Shard(
            tensor=shard_tensor,
            metadata=shard_metadata,
        )
    ]
    sharded_tensor = ShardedTensor._init_from_local_shards(
        local_shards,
        (2, 4),
    )

global_sharded_tensor_metadata = copy.deepcopy(rank_metadata)