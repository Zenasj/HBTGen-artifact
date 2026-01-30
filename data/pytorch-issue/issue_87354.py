import torch.nn as nn

import torch
import numpy as np
x = torch.rand(1, 32, 512, 512, 256)
m = torch.nn.Conv3d(32, 1, kernel_size=1, padding=0,stride=1,bias=False)
x = m(x)  # Assert!!
print(x)
numpy_x = x.cpu().detach().numpy()
print(np.where(numpy_x != 0), numpy_x.shape)

import torch
import numpy as np
x = torch.rand(1, 32, 512, 512, 256).to('cuda:0')
def test():
    tmp_result= torch.nn.LazyConv3d(out_channels=1, kernel_size=1, padding=0, stride=1, bias=False).to('cuda:0')
    return tmp_result
m = test()
x = m(x) 
print(x)
numpy_x = x.cpu().detach().numpy()
r_e_s=np.where(numpy_x != 0)
print(r_e_s)