import torch

self = torch.randn([1,1,1,1], dtype=torch.complex64)
other = torch.randn([1,1,1,1,2], dtype=torch.float64)

self.view_as(other)