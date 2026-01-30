import torch.nn as nn
import random

import torch
import numpy

y = numpy.random.rand(1,416,416,2).astype('float64')
y = numpy.transpose(y, (0,3,1,2))

filters = numpy.random.rand(3,3,2,16).astype('float64')
filterx = numpy.transpose(filters, (3,2,0,1))

x = torch.nn.Conv2d(2, 16, 3,  stride=1, padding=0, dilation=1, groups=1, bias=False) 
x.weight = torch.nn.Parameter(torch.from_numpy(filterx))
l = x(torch.from_numpy(y))
l = l.detach().numpy()

x.weight = torch.nn.Parameter(torch.from_numpy(filterx.astype('float32')))
t = x(torch.from_numpy(y.astype('float32')))
t = t.detach().numpy()

print(t.shape)
print(l.shape)
print(numpy.abs(t-l).max())

import torch
import numpy

y = numpy.random.rand(1,416,416,2).astype('float64')
y = numpy.transpose(y, (0,3,1,2))

filters = numpy.random.rand(3,3,2,16).astype('float64')
filterx = numpy.transpose(filters, (3,2,0,1))

x = torch.nn.Conv2d(2, 16, 3,  stride=1, padding=0, dilation=1, groups=1, bias=False)
x.weight = torch.nn.Parameter(torch.from_numpy(filterx))
l = x(torch.from_numpy(y))
l = l.detach().numpy()

x.weight = torch.nn.Parameter(torch.from_numpy(numpy.array(filterx.tolist()).astype('float32')))
t = x(torch.from_numpy(y.astype('float32')))
t = t.detach().numpy()

print(t.shape)
print(l.shape)
print(numpy.abs(t-l).max())