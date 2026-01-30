@runtime_checkable
class Shardable(Protocol):
    """
    Indicates whether a datapipe is shardable. A shardable datapipe distributes (shards) the data records
    amongst the dataloader worker instances in such a way that the same record is not processed twice
    when the dataloader ``num_workers > 1``.

    .. note::
      A shardable datapipe must implement BOTH the indicator method ``is_shardable()``
      AND the sharding logic ``apply_sharding(num_instances, instance_id)``.

    Use with ``isinstance()`` to check whether a datapipe is shardable

    .. doctest::

     >>> from torchdata.datapipes.iter import IterableWrapper
     >>> from ape.datapipes import Shardable
     >>> dp = IterableWrapper([1,2,3]).sharding_filter()
     >>> isinstance(dp, Shardable)
     True

    """

    def is_shardable(self) -> bool:  # pragma: no cover
        ...

    def apply_sharding(
        self, num_instances: int, instance_id: int
    ) -> None:  # pragma: no cover
        ...

def apply_sharding(self, num_instances, instance_id):
      self.global_world_size = num_instances # world_size * num_dataloader_workers
      self.global_rank = instance_id # dataloader_worker_rank * num_dataloader_workers + rank * world_size

# sharding would be done by 
shard_idx % self.global_world_size == self.global_rank