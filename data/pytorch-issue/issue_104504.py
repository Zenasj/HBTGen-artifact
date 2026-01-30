raise AssertionError(
                        "FSDP does not support changing the parameters between "
                        f"forward and backward for {self._sharding_strategy}"
                    )