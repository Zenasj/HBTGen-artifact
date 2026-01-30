import torch

x = torch.randn(2, 6, 7, 8).to_mkldnn()
weight = torch.randn(6, 12, 3, 3)
bias = torch.randn(12)
padding = (0, 0)
stride = (1, 1)
dilation = (1, 1)
transposed = True
output_padding = (0, 0)
groups = 1
inputs = [x, weight, bias, stride, padding, dilation, transposed, output_padding, groups]
output = torch.ops.aten.convolution(*inputs)
print(output.shape)