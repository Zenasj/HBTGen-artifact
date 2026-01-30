import time

import torch
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 64
SHAPE = (3, 224, 224)


class DebugDataset(Dataset):
    def __len__(self):
        return 6400

    def __getitem__(self, index):
        return torch.zeros(SHAPE)


def verbose_stack(samples):
    tic = time.perf_counter()
    batch = torch.stack(samples)
    print("tensor stacked: {:.3f} ms".format((time.perf_counter() - tic) * 1000))
    return batch

def batched(loader):
    """Helper function to collate batches outside the dataloader."""
    batch = []
    for sample in loader:
        batch.append(sample)
        if len(batch) >= BATCH_SIZE:
            # BUG: collation slow after the first data loader.
            # - the bug persists even when this collation is wrapped with a second data loader.
            yield verbose_stack(batch)
            batch = []


def main():
    print(" Fast ".center(40, "="))
    for images in DataLoader(
        DebugDataset(), num_workers=8, batch_size=64, collate_fn=verbose_stack
    ):
        assert images.shape[0] == BATCH_SIZE

    print(" Slow ".center(40, "="))
    for __ in batched(DataLoader(DebugDataset(), num_workers=8, batch_size=None)):
        assert images.shape[0] == BATCH_SIZE

if __name__ == "__main__":
    main()

from webdataset import WebDataset, WebLoader

dataset = WebDataset(url).shuffle(1000).batched(128)
slow_loader = WebLoader(dataset).unbatched().shuffle(1000).batched(128)
#                                                             ^
#                                                             |
#                                                    this would be slow