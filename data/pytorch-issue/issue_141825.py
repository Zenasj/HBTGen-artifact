class _DistWrapper:
    """
    This is a wrapper around PG that provides a series of features around object collectives.

    It works without distributed initialized, where most collectives turns into nops.

    All variants that take functions are exception robust, meaning that if one or more
    ranks raise errors, all ranks will observe those.
    """

    def __init__(
        self,
        group: Optional[dist.ProcessGroup],
        use_dist: bool,
        coordinator_rank: int,
    ):
        self.group = group
        self.use_dist = use_dist
        self.coordinator_rank = coordinator_rank
        if self.use_dist:
            self.rank = dist.get_rank(group)
            self.is_coordinator = self.rank == coordinator_rank
        else:
            self.rank = 0
            self.is_coordinator = True

def gather_object(self, object: T) -> Optional[List[T]]:
        """Implement functionality similar to c10d::gather_object but without distributed enabled."""
        if self.use_dist:
            gather_objs = (
                cast(List[T], [None] * dist.get_world_size(self.group))
                if self.is_coordinator
                else None
            )

            dist.gather_object(
                obj=object,
                object_gather_list=gather_objs if self.is_coordinator else None,
                dst=self.coordinator_rank,
                group=self.group,
            )
            result = gather_objs
        else:
            result = [object]
        return result