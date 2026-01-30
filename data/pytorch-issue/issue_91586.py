import torch
import numpy as np
arg_1_tensor = torch.neg(torch.rand([2, 1, 3, 4, 2], dtype=torch.float32))
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([4, 4], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.randint(-64,32,[4], dtype=torch.int32)
arg_3 = arg_3_tensor.clone()
try:
  res = torch.lu_solve(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))