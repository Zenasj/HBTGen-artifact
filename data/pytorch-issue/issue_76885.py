import torch.nn as nn

import torch
from torch import nn as nn

device = 'cuda:0'
cpu_device = 'cpu'
weight_elem = 10
weight_feature_size = 4
include_last_offset = True

embedding_sum = nn.EmbeddingBag(weight_elem, weight_feature_size, mode='sum', scale_grad_by_freq=False, include_last_offset=include_last_offset)
embedding_sum.weight.data = torch.Tensor([[-0.1117, -0.4966,  0.1631, -0.8817],
                                          [ 0.0539,  0.6684, -0.0597, -0.4675],
                                          [-0.2153,  0.8840, -0.7584, -0.3689],
                                          [-0.3424, -1.4020,  0.3206, -1.0219],
                                          [ 0.7988, -0.0923, -0.7049, -1.6024],
                                          [ 0.2891,  0.4899, -0.3853, -0.7120],
                                          [ 0.7667,  0.0190,  0.0220,  1.1532],
                                          [-0.3393,  0.1559,  0.8966, -0.2968],
                                          [-0.6857, -0.0496, -1.2485, -0.8509],
                                          [-0.7690, -1.5606, -0.5309,  0.2178]]).data
print("sum cpu")
input = torch.Tensor([1, 2, 4, 5, 4, 3, 2, 9], device=cpu_device).long()
offsets = torch.Tensor([0, 3, 6], device=cpu_device).long()
output = embedding_sum(input, offsets)
embedding_sum.zero_grad()

print("sum cuda")
input_cuda = input.to(device)
offsets_cuda = offsets.to(device)
embedding_sum = embedding_sum.to(device)
output_cuda = embedding_sum(input_cuda, offsets_cuda)
embedding_sum.zero_grad()

print('embedding_sum weight = ', embedding_sum.weight.cpu().data)
print('cpu output = ', output)
print('cuda output = ', output_cuda.to("cpu"))

input    [1, 2, 4, 5, 4, 3, 2, 9]
offsets  [0, 3, 6]