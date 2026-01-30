#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import cProfile, pstats

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)

    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      output = F.log_softmax(x, dim=1)

      return output

def fun_profile(inputs):
    net = inputs[0]
    cProfile.runctx('fun(net)', globals(), locals(),
                    'profile_{:d}.out'.format(inputs[1]))

def fun(net):
    random_data = torch.rand(
        (1, 1, 28, 28), device=torch.device('cpu'))
    net(random_data)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cpu')
    
    net = Net().eval()
    net.to(device)

    number_workers = 10
    start_time = datetime.datetime.now()
    if number_workers == 0:
        # no multiprocessing
        fun_profile([net, number_workers])
    else:
        inputs = [[net, number_workers]]*(number_workers)
        with torch.multiprocessing.Pool(number_workers) as pool:
            pool.map(fun_profile, inputs)

    print('Total time: ', datetime.datetime.now() - start_time)

    s = pstats.Stats(
        'profile_{:d}.out'.format(number_workers))
    s.strip_dirs().sort_stats('cumtime').print_stats('(forward)')