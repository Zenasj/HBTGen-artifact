import torch.nn as nn

import torch
import numpy as np
arg_1 = 1
arg_class = torch.nn.CTCLoss(arg_1,)
arg_2_0_tensor = torch.rand([50, 3, 15], dtype=torch.float64)
arg_2_0 = arg_2_0_tensor.clone()
arg_2_1_tensor = torch.randint(-512,32768,[3, 30], dtype=torch.int64)
arg_2_1 = arg_2_1_tensor.clone()
arg_2_2_tensor = torch.randint(-32,32,[3], dtype=torch.int64)
arg_2_2 = arg_2_2_tensor.clone()
arg_2_3_tensor = torch.randint(-2,32,[3], dtype=torch.int64)
arg_2_3 = arg_2_3_tensor.clone()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,arg_2_3,]
try:
  res = arg_class(*arg_2)
except Exception as e:
  print("Error:"+str(e))