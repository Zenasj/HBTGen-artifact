import torch.nn as nn

def conv(i, w):
	conv_stride = 2
	num_channels = i.size(0)
	return torch.nn.functional.conv1d(
		i.unsqueeze(0), w.repeat(num_channels, 1, 1),
        stride=conv_stride, groups=num_channels).squeeze(0)

torch.manual_seed(0)
i1 = torch.rand((2, 8024))*1e2
w1 = torch.rand(25)*1e2

conv1 = conv(i1, w1)  # (2, 4000)
conv2 = conv(i1[0,:].unsqueeze(0), w1)  # (1, 4000)

print('conv error %.16f' % ((conv1[0,:] - conv2[0,:]).abs().max()))

def conv(i, w):
	conv_stride = 2
	num_channels = i.size(0)
	return torch.nn.functional.conv1d(
		i.unsqueeze(0), w.repeat(num_channels, 1, 1),
        stride=conv_stride, groups=num_channels).squeeze(0)

input1 = torch.tensor([
        [49.6257, 76.8222,  8.8477, 13.2030, 30.7423, 63.4079, 49.0093, 89.6445,
         45.5628, 63.2306, 34.8893, 40.1717,  2.2326, 16.8859, 29.3888, 51.8522,
         69.7668, 80.0011, 16.1029, 28.2269],
        [68.1609, 91.5194, 39.7100, 87.4156, 41.9408, 55.2907, 95.2738,  3.6165,
         18.5231, 37.3417, 30.5100, 93.2000, 17.5910, 26.9834, 15.0680,  3.1720,
         20.8130, 92.9799, 72.3109, 74.2336]])
input2 = input1[0, :].unsqueeze(0)
weight = torch.tensor([52.6296, 24.3658])
conv1 = conv(input1, weight)  # (2, 10)
conv2 = conv(input2, weight)  # (1, 10)
print('conv1', conv1)
print('conv2', conv2)
print('conv error %.16f' % ((conv1[0,:] - conv2[0,:]).abs().max()))

torch.manual_seed(0)
i1 = torch.rand((2, 20))*1e5
w1 = torch.rand(2)*1e5

i1 = i1.round()
w1 = w1.round()
conv1 = conv(i1, w1)  # (2, 10)
conv2 = conv(i1[0,:].unsqueeze(0), w1)  # (1, 10)

print('conv error %.16f' % ((conv1[0,:] - conv2[0,:]).abs().max()))

import torch

def conv(i, w):
    conv_stride = 2
    num_channels = i.size(0)
    return torch.nn.functional.conv1d(
            i.unsqueeze(0), w.repeat(num_channels, 1, 1),
            stride=conv_stride, groups=num_channels).squeeze(0)

torch.manual_seed(0)
i1 = torch.rand((2, 8024))*1e2
w1 = torch.rand(25)*1e2
i2 = i1.cuda()
w2 = w1.cuda()

# CPU path 
conv1 = conv(i1, w1)  # (2, 4000)
conv2 = conv(i1[0,:].unsqueeze(0), w1)  # (1, 4000)
print('cpu conv error %.16f' % ((conv1[0,:] - conv2[0,:]).abs().max()))

# GPU path
conv11 = conv(i2, w2)  # (2, 4000)
conv22 = conv(i2[0,:].unsqueeze(0), w2)  # (1, 4000)
print('gpu conv error %.16f' % ((conv11[0,:] - conv22[0,:]).abs().max()))

#compare cpu and gpu path
print("compare cpu and gpu result")
print('first conv error %.16f' % ((conv11 - conv1.cuda()).abs().max()))
print('second conv error %.16f' % ((conv22 - conv2.cuda()).abs().max()))