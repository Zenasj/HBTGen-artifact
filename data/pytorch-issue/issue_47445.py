#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader, IterableDataset

data_size = 100
batch_size = 5
worker_size = 5


class MyIterDataset(IterableDataset):
    def __iter__(self):
        for i in range(data_size):
            yield torch.ones([10])


def test_bug():
    loader = DataLoader(MyIterDataset(),
                        batch_size=batch_size,
                        drop_last=True,
                        num_workers=worker_size,
                        pin_memory=True,
                        persistent_workers=True)
    for epoch in range(2):
        print("EPOCH", epoch)
        for batch in loader:
            pass
    print("DONE")


if __name__ == "__main__":
    test_bug()