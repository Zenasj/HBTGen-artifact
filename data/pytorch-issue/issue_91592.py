import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.rand([4, 3, 4, 5], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([5, 4, 3, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([5], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 1
arg_4_1 = 1
arg_4_2 = 1
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5_0 = 0
arg_5_1 = 0
arg_5_2 = 0
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
arg_6_0 = 1
arg_6_1 = 1
arg_6_2 = 1
arg_6 = [arg_6_0,arg_6_1,arg_6_2,]
arg_7 = 0
try:
  res = torch.nn.functional.conv3d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  print("Error:"+str(e))

import torch
import numpy as np
arg_1_tensor = torch.randint(-1024,4,[3, 1, 3, 3], dtype=torch.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([1, 1, 3, 3, 3], dtype=torch.complex64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([1], dtype=torch.complex64)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 1
arg_4_1 = 1
arg_4_2 = 1
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5_0 = 0
arg_5_1 = 0
arg_5_2 = 0
arg_5 = [arg_5_0,arg_5_1,arg_5_2,]
arg_6_0 = 1
arg_6_1 = 1
arg_6_2 = 1
arg_6 = [arg_6_0,arg_6_1,arg_6_2,]
arg_7 = 0
try:
  res = torch.nn.functional.conv3d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  print("Error:"+str(e))