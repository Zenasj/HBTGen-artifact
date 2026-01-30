py
import torch

class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10000000

    def __getitem__(self, any):
        return torch.empty(0)

if __name__ == '__main__':
    dl = torch.utils.data.DataLoader(
        Dataset(),
        batch_size=40960,
        num_workers=1)

    it = iter(dl)

    for i, x in enumerate(it):
        print(x.shape)
        raise RuntimeError()

py
import torch


class BigSampler(object):
    def __iter__(self):
        for idx in range(20):
            yield [idx for idx in range(40960)]


class Dataset(torch.utils.data.Dataset):

    def __init__(self):
        self.first_time = True

    def __len__(self):
        return 10000000

    def __getitem__(self, indices):
        # NB: this is called *once* per dataloader fetch since we disabled auto-batching!
        if not self.first_time:
            # simulate some work so that the index_queue is likely non-empty 
            # (and thus the main process's queue putting is waiting). 
            import time
            time.sleep(0.1)
        self.first_time = False
        return torch.empty(0)


if __name__ == '__main__':
    dl = torch.utils.data.DataLoader(
        Dataset(),
        batch_size=None,
        sampler=BigSampler(),
        num_workers=1)

    it = iter(dl)

    for i, x in enumerate(it):
        print(x.shape)
        raise RuntimeError()