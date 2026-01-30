import torch

py
from torch.utils.data.dataloader_experimental import DataLoader2

from torchdata.datapipes.iter import IterableWrapper, IterDataPipe, Shuffler


class Sorter(IterDataPipe):
    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        return iter(sorted(self.datapipe))


data = list(range(1000))
dp = IterableWrapper(data)
dp = Shuffler(dp).set_shuffle(False)
dp = Sorter(dp)

dl2 = DataLoader2(dp, shuffle=True, batch_size=None)

assert list(dl2) == data  # fails unless you hit a lucky random seed