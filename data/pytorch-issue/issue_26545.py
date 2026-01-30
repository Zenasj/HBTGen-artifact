class ChunkDataReader(object):
    def __init__(self):
        r"""The reader is initialized here"""

    def __call__(self, index):
        r"""Returns `list(samples)` for a given :attr:`index"""

class DistributedChunkSampler(Sampler):
    def __init__(self, num_replicas, rank=0, num_chunks=0, shuffle=False):
        r"""Returns a new DistributedChunkSampler instance"""

    def set_rank(self, rank):
        r"""Set rank for the current sampler instance"""

    def set_epoch(self, epoch):
        r"""Set epoch for the current sampler instance"""

class ChunkDataset(IterableDataset):
    r"""Dataset which uses hierarchical sampling"""

    def __init__(self, chunk_sampler, chunk_reader, shuffle_cache=True):
        r"""Returns a new ChunkDataset instance"""

    def __iter__(self):
        r"""Returns an Iterator for batching"""

    def __next__(self):
        r"""Returns the next value in the Iterator"""

    def reset(self, epoch=None):
        r"""Resets internal state of ChunkDataset"""