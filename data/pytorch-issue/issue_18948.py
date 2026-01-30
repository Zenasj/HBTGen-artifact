import torch.nn as nn

import torch
import torch.nn.functional as F

x = torch.Tensor([[[[0., 2., 2.],
                    [0., 0., 1.],
                    [2., 0., 0.]]]])
w = torch.Tensor([[[[1., 1., 0.],
                    [1., 0., 3.],
                    [0., 3., 1.]]]])

x1 = x[:,:,:,2:] # Using only the last column from x, containing [2,1,0]
w1 = w[:,:,:,2:] # Using only the last column from w, containing [0,3,1]
#The result should be ((2*0) + (1*3) + (0*1)) = 3

print(F.conv2d(x1, w1, None, padding=0, stride=1)) # tensor([[[[1.]]]]) WRONG!!!!
print(F.conv2d(x1.cuda(), w1.cuda(), None, padding=0, stride=1)) # tensor([[[[3.]]]], device='cuda:0') as expected