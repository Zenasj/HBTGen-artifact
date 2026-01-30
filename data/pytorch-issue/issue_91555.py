import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.neg(torch.rand([1, 2, 4, 6, 3], dtype=torch.float64))
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-1,32,[1, 2, 4, 6, 3], dtype=torch.int64)
arg_2 = arg_2_tensor.clone()
arg_3 = 3
arg_4_0 = 2
arg_4_1 = 1
arg_4_2 = 2
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5 = 1
arg_6_0 = 1
arg_6_1 = 2
arg_6_2 = 6
arg_6_3 = 6
arg_6_4 = 5
arg_6 = [arg_6_0,arg_6_1,arg_6_2,arg_6_3,arg_6_4,]
try:
  res = torch.nn.functional.max_unpool3d(arg_1,arg_2,kernel_size=arg_3,stride=arg_4,padding=arg_5,output_size=arg_6,)
except Exception as e:
  print("Error:"+str(e))