import torch
import torch.nn as nn

input_example = torch.zeros(1,1,3,3)
# set the middle pixel to be a one (the rest are zeros)
input_example[:,:,1,1] = 1.

conv = nn.Conv2d(1, 5, kernel_size=3, bias=False)
# set the conv weights to be five 3x3 identity matrices

conv.weight.data = torch.eye(3,3).unsqueeze(0).repeat(5,1,1,1)
print(conv(input_example))