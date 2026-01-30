batch_size_per_feature = list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, local_split) for x in batch_size_per_rank
                )
            )