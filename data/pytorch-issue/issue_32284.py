import torch.nn as nn

3
import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from torch import nn
from memory_profiler import profile
import resource

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.maxp1 = nn.MaxPool2d(1, 1)

    def forward(self, x):
        return self.maxp1(x)

def debug_memory():
    print('maxrss = {}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

@profile
def func():
    batch = 2
    channels = 1
    side_dim = 80
    model = Network()
    x = torch.randn([batch, channels, side_dim, side_dim])
    while True:
        y = model(x)
        debug_memory()

import os
import torch
torch.set_num_threads(1)
from torch import nn
from memory_profiler import profile
import resource
import gc


def debug_memory():
    val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print('maxrss = {}'.format(val))
    return val

def fn(x):
    # choose one
    # torch._C._nn.max_pool2d_with_indices(x, 1)
    torch.relu(x)

@profile
def func():
    batch = 2
    channels = 1
    side_dim = 80
    x = torch.randn([batch, channels, side_dim, side_dim])
    old_val = debug_memory()
    for i in range(1000):
        fn(x)
        new_val = debug_memory()
        if new_val != old_val:
            print("Increased at {}: {}".format(i, new_val - old_val))
            old_val = new_val

func()

import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn.functional as F
from torch import nn
from memory_profiler import profile
import resource

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.maxp1 = nn.MaxPool2d(1, 1)

    def forward(self, x):
        #return torch.relu(x)
        #return torch.threshold(x, 0, 0)
        #return torch._C._nn.log_sigmoid(x)
        return torch._C._nn.max_pool2d_with_indices(x, 1)
        #return self.maxp1(x)

def debug_memory():
    print('maxrss = {}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

@profile
def func():
    batch = 2
    channels = 1
    side_dim = 80
    model = Network()
    x = torch.randn([batch, channels, side_dim, side_dim])
    for i in range(100000):
        y = model(x)
        if i % 1000 == 0:
            debug_memory()

if __name__ == '__main__':
    func()