import torch.nn as nn

import torch
coords = torch.tensor([[[[-1., -1.],
                         [ 1., -1.]],

                        [[-1.,  1.],
                         [ 1.,  1.]]]])

im = torch.zeros([1, 1, 32769, 65536])

result = torch.nn.functional.grid_sample(im, coords, align_corners=False)

import torch
coords = torch.tensor([[[[-1., -1.],
                         [ 1., -1.]],

                        [[-1.,  1.],
                         [ 1.,  1.]]]])

im = torch.zeros([1, 1, 32768, 65536])

result = torch.nn.functional.grid_sample(im, coords, align_corners=False)