import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.rand([2, 2, 14, 8], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([2, 3, 4, 4], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([3], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 1
arg_4_1 = 1
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 2
arg_5_1 = 2
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = 0
arg_6_1 = 0
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 0
arg_8_0 = 3
arg_8_1 = 3
arg_8 = [arg_8_0,arg_8_1,]
try:
  res = torch.nn.functional.conv_transpose2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,arg_8,)
except Exception as e:
  print("Error:"+str(e))