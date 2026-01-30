import torch
import numpy as np
arg_1_tensor = torch.rand([2, 2], dtype=torch.complex64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-64,4096,[], dtype=torch.int32)
arg_2 = arg_2_tensor.clone()
arg_3 = True
try:
  res = torch.lu_unpack(arg_1,arg_2,unpack_pivots=arg_3,)
except Exception as e:
  print("Error:"+str(e))