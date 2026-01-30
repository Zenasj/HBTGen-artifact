import torch
from torch.nn import Conv2d

torch.manual_seed(13009)        # Fixed random number seed

ci = 4              # channel in  : This problem only occurs when the number of input and output channels is greater than 1
co = 3              # channel out : This problem only occurs when the number of input and output channels is greater than 1
b = 2               # batch size, if change this, you need to synchronize and modify the following code.
h = 8               # feature's high
w = 6               # feature's wide
k = 3               # kernel size
device = "cpu"      # problem only occurs when use cpu

c2d = Conv2d(ci, co, k, 1, 0, 1, device=device)     # a simple net

# # If you need to manually set the convolution core weights
# w1 = torch.randn(co, ci, k,k)
# b1 = torch.randn(co)
# c2d.weight.data = w1
# c2d.bias.data = b1

for i in range(10):     # Multiple tests
    x = torch.randn(2, ci, h, w, device=device)
    y1 = c2d(x)                                                             # One batch in
    y2 = torch.cat([c2d(x[0:1, :, :, :]), c2d(x[1:2, :, :, :])], dim=0)     # One by one
    diff = torch.sum(torch.abs(y2 - y1), dim=[1, 2, 3])                     # the different sum of feature per batch
    print(diff)

ci = 4              # channel in  : This problem only occurs when the number of input and output channels is greater than 1
co = 3              # channel out : This problem only occurs when the number of input and output channels is greater than 1
b = 2               # batch size, if change this, you need to synchronize and modify the following code.
h = 8               # feature's high
w = 6               # feature's wide
k = 3               # kernel size
device = "cpu"      # problem only occurs when use cpu
dtype = torch.float32

c2d = Conv2d(ci, co, k, 1, 0, 1, device=device, dtype=dtype)     # a simple net

for i in range(10):     # Multiple tests
    x = torch.randn(2, ci, h, w, device=device, dtype=dtype)
    y1 = c2d(x)                                                             # One batch in
    y2 = torch.cat([c2d(x[0:1, :, :, :]), c2d(x[1:2, :, :, :])], dim=0)     # One by one
    diff = torch.sum(torch.abs(y2 - y1), dim=[1, 2, 3])                     # the different sum of feature per batch
    print(diff)