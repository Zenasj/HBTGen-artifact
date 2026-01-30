import torch.nn as nn

import torch

pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = torch.nn.MaxUnpool2d(2, stride=2)
input = torch.torch.tensor([[[[ 1.,  2,  3,  4, 0],
                              [ 5,  6,  7,  8, 0],
                              [ 9, 10, 11, 12, 0],
                              [13, 14, 15, 16, 0],
                              [0, 0, 0, 0, 0]]]])
output, indices = pool(input)
unpooled55 = unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
torch.testing.assert_allclose(pool(unpooled55)[0], output)