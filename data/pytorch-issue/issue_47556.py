import torch.nn as nn
import random

import torch
import numpy as np

def create_tensor(shape, dtype, min=0, max=100):
    data = np.random.uniform(low=min,high=max,size=shape).astype(dtype)
    tensor = torch.from_numpy(data)
    return tensor

shape = (2,3,11,9)
tensor = create_tensor(shape, np.float32, 0, 100)
ksize = (2,2)
stride = (2,2)
padding = (1,1)
dilation = 1
ceil_mode = False
output1 = torch.nn.quantized.functional.max_pool2d(tensor, ksize, stride, padding, dilation, ceil_mode)
ceil_mode = True
output2 = torch.nn.quantized.functional.max_pool2d(tensor, ksize, stride, padding, dilation, ceil_mode)
print(output1.shape)
print(output2.shape)