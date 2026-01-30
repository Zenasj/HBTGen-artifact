import torch
import numpy as np
arg_1_tensor = torch.rand([3], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.neg(torch.rand([], dtype=torch.float32))
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.neg(torch.rand([3, 3, 3], dtype=torch.float16))
arg_3 = arg_3_tensor.clone()
try:
  res = torch.lt(arg_1,arg_2,out=arg_3,)
except Exception as e:
  print("Error:"+str(e))