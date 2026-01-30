import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.rand([1, 3, 4, 4, 4], dtype=torch.complex64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([3, 3, 3, 3, 3], dtype=torch.complex64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([3], dtype=torch.complex64)
arg_3 = arg_3_tensor.clone()
arg_4_0 = 2
arg_4_1 = 2
arg_4_2 = 2
arg_4 = [arg_4_0,arg_4_1,arg_4_2,]
arg_5 = 2
arg_6_0 = 1
arg_6_1 = 1
arg_6_2 = 1
arg_6 = [arg_6_0,arg_6_1,arg_6_2,]
arg_7 = 0
try:
  res = torch.nn.functional.conv_transpose3d(arg_1,arg_2,arg_3,stride=arg_4,padding=arg_5,output_padding=arg_6,groups=arg_7,)
except Exception as e:
  print("Error:"+str(e))