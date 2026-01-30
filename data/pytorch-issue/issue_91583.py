import torch.nn as nn

import torch
import numpy as np
arg_1_0 = 4050212686
arg_1 = [arg_1_0,]
arg_2 = 0.001
arg_class = torch.nn.LayerNorm(arg_1,arg_2,)
arg_3_0_tensor = torch.rand([4, 5, 5], dtype=torch.float64)
arg_3_0 = arg_3_0_tensor.clone()
arg_3 = [arg_3_0,]
try:
  res = arg_class(*arg_3)
except Exception as e:
  print("Error:"+str(e))