import random

import os
import subprocess
import numpy as np
import gc
import torch
import torch.nn as nn
from memory_profiler import profile
from torch.utils.data import Dataset, DataLoader


def current_memory_usage():
    '''Returns current memory usage (in MB) of a current process'''
    out = subprocess.Popen(['ps', '-p', str(os.getpid()), '-o', 'rss'],
                           stdout=subprocess.PIPE).communicate()[0].split(b'\n')
    mem = float(out[1].strip()) / 1024
    return mem


class RandomDataset(Dataset):

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        width = np.random.randint(200, 401)
        height = np.random.randint(200, 401)
        image = np.random.rand(3, width, height)
        return image, np.random.randint(0, 2)


def padding_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    max_width = max(img.shape[1] for img in images)
    max_height = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        right_pad = max_width - img.shape[1]
        bottom_pad = max_height - img.shape[2]
        padding_tuple = ((0, 0), (0, right_pad), (0, bottom_pad))
        padded_img = np.pad(img, padding_tuple)
        padded_images.append(torch.from_numpy(padded_img.astype(np.float32)))

    images_batch = torch.stack(padded_images)
    labels_batch = torch.stack([torch.from_numpy(np.array(lab)) for lab in labels])
    return images_batch, labels_batch


class BasicModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


@profile
def main():
    random_ds = RandomDataset(n_samples=100)
    input_pipe = DataLoader(random_ds, batch_size=2, collate_fn=padding_fn)
    model = BasicModel()

    for epoch in range(1000):
        for i_step, (batch_images, batch_labels) in enumerate(input_pipe):
            model(batch_images)

            print('{}-{}:  \t\tMemory usage: {:.1f}MB'.format(epoch, i_step, current_memory_usage()))
            gc.collect()


if __name__ == "__main__":
    main()

def padding_fn(batch):
    images_batch = torch.rand((2, 3, 100, 100))
    labels_batch = torch.randint(0, 2, (2,))
    return images_batch, labels_batch

from torch.utils import mkldnn
mkldnn.enabled = False