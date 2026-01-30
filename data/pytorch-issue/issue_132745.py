def recursive_undo(self, sched=None):
        # recursively undo any step performed by the initialisation of
        # schedulers
        scheds = self if sched is None else sched

        if hasattr(scheds, "_schedulers"):
            for s in scheds._schedulers:
                self.recursive_undo(s)
        elif hasattr(scheds, "last_epoch"):
            scheds.last_epoch -= 1