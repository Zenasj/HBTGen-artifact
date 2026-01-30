import torch
import numpy as np
import random

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

class MNISTCSVChunkDataReader(torch.utils.data.ChunkDataReader):
    r"""Reads chunk of MNIST CSV data for the specified chunk index."""

    def __init__(self, chunk_files):
        super(MNISTCSVChunkDataReader, self).__init__()
        assert isinstance(chunk_files, list), 'chunk_files must be a `list`'
        assert len(chunk_files) > 0, 'chunk_files must contain at least one chunk'

        self.chunk_files = chunk_files

    def __call__(self, index):
        r"""
        Returns a `tuple(data, target)` or `None`, where
        `data` and `target` are `numpy.ndarray((batch_size, actual_data))`
        """

        assert isinstance(index, int), 'index must be a `int`'
        assert index < len(
            self.chunk_files), 'index must be < `len(chunk_files)`'

        csv = pd.read_csv(self.chunk_files[index])
        data = csv.loc[:, csv.columns != "label"].values.astype(np.uint8).reshape(-1, 28, 28)
        target = csv.label.values.astype(np.uint8)

        return list(zip(data, target))

for _ in range(epochs):
    random.shuffle(datasets)
    for data_ in torch.utils.data.DataLoader(ChunkDataset(datasets), num_workers=args.workers):
        data = collate_fn(data_)
        train(args, model, device, train_loader, optimizer, epoch, ...)
        test(args, model, device, test_loader, ...)