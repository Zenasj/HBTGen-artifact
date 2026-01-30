import torch.nn as nn

import torch
import numpy as np
arg_1 = 3
arg_2 = 68
arg_3 = 23
arg_class = torch.nn.ConvTranspose3d(arg_1, arg_2, kernel_size=arg_3,)
arg_4_0_tensor = torch.rand([8, 3, 16, 16, 16], dtype=torch.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
try:
    res = arg_class(*arg_4)
except Exception as e:
    print("Error:"+str(e))

import torch
import numpy as np
import time
arg_1 = 3
arg_2 = 68
arg_3 = 23
arg_class = torch.nn.ConvTranspose3d(arg_1, arg_2, kernel_size=arg_3,)
arg_4_0_tensor = torch.rand([8, 3, 16, 16, 16], dtype=torch.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
st = time.time()
try:
    res = arg_class(*arg_4)
except Exception as e:
    print("Error:"+str(e))
print(time.time() - st)
print(res.shape)

60.098710775375366
torch.Size([8, 68, 38, 38, 38])