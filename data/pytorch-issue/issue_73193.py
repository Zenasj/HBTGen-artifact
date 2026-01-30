import torch

self = torch.full((1, 1, 1, 24000,), 1, dtype=torch.float32, requires_grad=False)
weight = torch.full((1, 1, 1, 24000,), 1, dtype=torch.float32, requires_grad=False)
bias = torch.full((1, 1, 1, 24000,), 1, dtype=torch.float32, requires_grad=False)
padding = [0, 0]
stride = [0, 0]
dilation = [0, 0]
groups = 0
torch.mkldnn_convolution(self, weight, bias, padding, stride, dilation, groups)