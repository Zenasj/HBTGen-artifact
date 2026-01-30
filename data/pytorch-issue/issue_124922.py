import os
import time

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def fast_collate(batch):
    images = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = images[0].size[0]
    h = images[0].size[1]
    tensor = torch.zeros((len(images), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(images):
        np_array = np.asarray(img, dtype=np.uint8)
        if np_array.ndim < 3:
            np_array = np.expand_dims(np_array, axis=-1)
        np_array = np.rollaxis(np_array, 2)
        tensor[i] += torch.from_numpy(np_array.copy())

    return tensor, targets


def load():
    train_dir = os.path.join("/data/imagenet2012", "train")
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]))
    train_sampler = None
    batch_size = 512
    workers = 64
    train_loader = MultiEpochsDataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler,
        collate_fn=fast_collate, drop_last=True)

    start_time = time.time()
    previous_timestamp = start_time
    for i, (images, target) in enumerate(train_loader):
        print(f"Step {i} : {time.time() - previous_timestamp}")
        previous_timestamp = time.time()
        if i == 29:
            print(f"Step Avg : {(previous_timestamp - start_time) / 30}")
            break


if __name__ == "__main__":
    load()