import torch
while True:
    x = torch.randn(10)
    x = x.refine_names('batch')

import torch
x = torch.randn(10)
while True:
    x = x.refine_names('batch')

import torch
x = torch.randn(10)
while True:
    x.refine_names('batch')

import torch
while True:
    x = torch.randn(10, names=['batch'])