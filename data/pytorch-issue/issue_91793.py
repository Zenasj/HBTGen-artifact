import torch.nn as nn

import torch
import numpy as np
arg_1 = 3431068031
arg_2 = 0.001
arg_3 = 0.3
arg_4 = False
arg_class = torch.nn.BatchNorm3d(arg_1,arg_2,arg_3,arg_4,)
arg_5_0_tensor = torch.rand([0, 5, 2, 2, 2], dtype=torch.float64)
arg_5_0 = arg_5_0_tensor.clone()
arg_5 = [arg_5_0,]
try:
  res = arg_class(*arg_5)
except Exception as e:
  print("Error:"+str(e))