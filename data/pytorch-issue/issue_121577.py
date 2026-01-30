class ChainedScheduler(LRScheduler):
    def __init__(self, schedulers: Sequence[LRScheduler]):
        if len(schedulers) < 1:
            raise ValueError("ChainedScheduler expects at least one scheduler to be chained, but got no schedulers.")

        first_scheduler: LRScheduler = None
        for scheduler in schedulers:
            if not isinstance(scheduler, LRScheduler):
                raise TypeError(
                    f"ChainedScheduler expects all schedulers to be of type LRScheduler, "
                    f"but got {type(scheduler)}"
                )
            if first_scheduler is None:
                first_scheduler = scheduler

            else:
                if first_scheduler.optimizer != scheduler.optimizer:
                    raise ValueError(
                        "ChainedScheduler expects all schedulers to belong to the same optimizer, but "
                        f"got schedulers at index {0} and {scheduler} to be different"
                    )
        self._schedulers = schedulers
        self.optimizer = first_scheduler.optimizer

class ChainedScheduler(LRScheduler):
    def __init__(self, schedulers: Sequence[LRScheduler], optimizer: Optional[Optimizer] = None):
        if len(schedulers) < 1:
            raise ValueError("ChainedScheduler expects at least one schedulers to be chained, but got no schedulers.")

        optimizer = optimizer or schedulers[0].optimizer
        for sch_idx, scheduler in enumerate(schedulers):
            if not isinstance(scheduler, LRScheduler):
                raise TypeError(
                    f"ChainedScheduler expects all schedulers to be of type LRScheduler, "
                    f"but got {type(scheduler)}"
                )

            if not sch_idx and optimizer is None:
                optimizer = scheduler.optimizer

            if optimizer != scheduler.optimizer:
                raise ValueError(
                    "ChainedScheduler expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler {scheduler.__class__.__name__} at index {sch_idx} has {scheduler.optimizer}, "
                    f"which is different from {optimizer.__class__.__name__}."
                )
        self._schedulers = schedulers
        self.optimizer = optimizer