import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.rand([1, 3, 5, 7], dtype=torch.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 2
arg_2_1 = 2
arg_2 = [arg_2_0,arg_2_1,]
arg_3 = None
arg_4_0 = 0.5
arg_4_1 = 0.5
arg_4 = [arg_4_0,arg_4_1,]
arg_5 = False
arg_6_tensor = torch.tensor([], dtype=torch.float64)
arg_6 = arg_6_tensor.clone()
try:
  res = torch.nn.functional.fractional_max_pool2d(arg_1,arg_2,arg_3,arg_4,arg_5,_random_samples=arg_6,)
except Exception as e:
  print("Error:"+str(e))