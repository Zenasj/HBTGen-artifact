import torch
import numpy as np
arg_1_tensor = torch.neg(torch.rand([2, 2, 5, 5], dtype=torch.complex64))
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-128,512,[2, 2, 5], dtype=torch.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.rand([2, 2, 5, 1], dtype=torch.complex64)
arg_3 = arg_3_tensor.clone()
arg_4 = True
try:
  res = torch.linalg.ldl_solve(arg_1,arg_2,arg_3,hermitian=arg_4,)
except Exception as e:
  print("Error:"+str(e))