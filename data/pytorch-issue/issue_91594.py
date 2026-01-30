import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.rand([5, 1, 224], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([2, 1, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3 = None
arg_4_0 = 2
arg_4 = [arg_4_0,]
arg_5_0 = 1
arg_5 = [arg_5_0,]
arg_6_0 = 1
arg_6 = [arg_6_0,]
arg_7 = 0
try:
  res = torch.nn.functional.conv1d(arg_1,arg_2,arg_3,arg_4,arg_5,arg_6,arg_7,)
except Exception as e:
  print("Error:"+str(e))

import torch
import numpy as np
arg_1_tensor = torch.neg(torch.rand([2, 4], dtype=torch.bfloat16))
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([2, 2, 3], dtype=torch.bfloat16)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.neg(torch.rand([3, 3, 3], dtype=torch.complex128))
arg_3 = arg_3_tensor.clone()
arg_4 = 3
arg_5 = 1
arg_6 = 0
arg_7 = 2
try:
  res = torch.nn.functional.conv1d(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,groups=arg_6,dilation=arg_7,)
except Exception as e:
  print("Error:"+str(e))

import torch
import numpy as np
arg_1_tensor = torch.randint(-4,1,[2, 4, 8], dtype=torch.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-4096,16,[2, 2, 3], dtype=torch.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.randint(-32768,16,[2], dtype=torch.int64)
arg_3 = arg_3_tensor.clone()
arg_4 = 31
arg_5 = 20
arg_6 = 0
arg_7 = 61
try:
  res = torch.nn.functional.conv1d(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,groups=arg_6,dilation=arg_7,)
except Exception as e:
  print("Error:"+str(e))