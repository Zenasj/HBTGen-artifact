import torch
import torch.nn as nn

my_func = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2))).cuda()
my_input = torch.randn(8, 128, 128, 80).cuda()
print(my_func(my_input).shape)