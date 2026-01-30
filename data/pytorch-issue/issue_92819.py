import torch
import numpy as np
arg_1_tensor = torch.neg(torch.rand([2], dtype=torch.float64))
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.tensor([], dtype=torch.bool)
arg_2 = arg_2_tensor.clone()
try:
  res = torch.linalg.inv(arg_1,out=arg_2,)
except Exception as e:
  print("Error:"+str(e))