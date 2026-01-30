import torch.nn as nn

import random

import torch

import psutil


def conv2d_forever():
    tot_batch = 0

    while True:
        with torch.no_grad():
            # torch.nn.functional.conv2d(torch.ones(1, 3, random.randint(8, 64), random.randint(8, 256)), torch.randn(3, 1, 3, 3), groups=3)
            torch.nn.functional.conv2d(torch.ones(1, 3, random.randint(8, 64), random.randint(8, 256)), torch.randn(3, 3, 3, 3))

        # sample = blur(sample)

        mem = psutil.virtual_memory()
        if tot_batch % 1000  == 0:
            print(
                f'{tot_batch:8} - {mem.free / 1024 ** 3:10.2f} - {mem.available / 1024 ** 3:10.2f} - {mem.used / 1024 ** 3:10.4f}')

        tot_batch += 1


if __name__ == '__main__':
    conv2d_forever()