import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.rand([4, 8, 224, 224], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([4, 2, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([4], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 2
arg_4_1 = 2
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 1
arg_5_1 = 1
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = 1
arg_6_1 = 1
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 0
try:
  res = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  print("Error:"+str(e))

import torch
import numpy as np
arg_1_tensor = torch.rand([4, 6, 5], dtype=torch.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([6, 2, 3, 2], dtype=torch.float64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.neg(torch.rand([], dtype=torch.float64))
arg_3 = arg_3_tensor.clone()
arg_4 = 0
try:
  res = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,groups=arg_4,)
except Exception as e:
  print("Error:"+str(e))

import torch
import numpy as np
arg_1_tensor = torch.rand([1, 3, 224, 224], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([64, 3, 7, 7], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([64], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 2
arg_4_1 = 2
arg_4 = [arg_4_0,arg_4_1,]
arg_5_0 = 3
arg_5_1 = 3
arg_5 = [arg_5_0,arg_5_1,]
arg_6_0 = 44
arg_6_1 = 13
arg_6 = [arg_6_0,arg_6_1,]
arg_7 = 0
try:
  res = torch.nn.functional.conv2d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  print("Error:"+str(e))