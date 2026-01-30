import torch.nn as nn
import random

'''
Demostrates what seem to be a performance bug (or puzzle) in PyTorch:
conv2d operation on trained weights is 20-50x slower than using random weights
'''

import contextlib
import torch
from torch import nn
import time
import contextlib
import urllib.request
import os

if not torch.set_flush_denormal(True):
    print("Unable to set flush denormal")
    print("Pytorch compiled without advanced CPU")
    print("at: https://github.com/pytorch/pytorch/blob/84b275b70f73d5fd311f62614bccc405f3d5bfa3/aten/src/ATen/cpu/FlushDenormal.cpp#L13")

def download_as(url, filename):
    if os.path.exists(filename):
        return  # do not download twice
    print('Downloading', url, 'as', filename)
    data = urllib.request.urlopen(url).read()
    with open(filename, 'wb') as f:
        f.write(data)

@contextlib.contextmanager
def timeit(message=''):
    start = time.time()
    try:
        yield
    finally:
        print(message, 'Time elapsed: ', (time.time() - start))

def printit(header, x):
    print(header)
    print('\tshape:', x.shape)
    print('\tdtype:', x.dtype)
    print('\tany nans?:', torch.isnan(x).any().item())
    print('\tmin:', x.min().item())
    print('\tmax:', x.max().item())
    print('\tmean:', x.mean().item())
    print('\tstd:', x.std().item())


# 18Mb file with trained weights:
download_as(
    'https://drive.google.com/uc?export=download&id=15Z6ACJKRep9Wjl_heyG1WS34Vsb_IUZn',
    'trained_weights_numpy.pkl'
)
with open('trained_weights_numpy.pkl', 'rb') as f:
    import pickle
    z = pickle.load(f)
weight = torch.tensor(z)
printit('Trained weight', weight)

# random convolution weights
weight_random = torch.zeros(512, 1024, 3, 3)
weight_random.normal_(std=0.001)
printit('Good (random) weight', weight_random)

# random input vector
x = torch.zeros(1, 1024, 32, 32)
x.normal_()
printit('Input vector', x)

with timeit('Fast conv2d (random weights)'):
    torch.nn.functional.conv2d(x, weight_random)

with timeit('Slow convd (trained weights)'):
    torch.nn.functional.conv2d(x, weight)