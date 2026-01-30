import torch
import numpy as np
arg_1_tensor = torch.rand([2, 1, 1], dtype=torch.complex128)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([2, 1, 1], dtype=torch.complex128)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.randint(-32768,1,[2, 1], dtype=torch.int32)
arg_3 = arg_3_tensor.clone()
try:
  res = torch.lu_solve(arg_1,arg_2,arg_3,)
  print(res)
except Exception as e:
  print("Error:"+str(e))