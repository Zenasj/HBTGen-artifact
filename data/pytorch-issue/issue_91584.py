import torch
import numpy as np
arg_1_tensor = torch.rand([0, 1, 4, 6, 7], dtype=torch.float64)
arg_1 = arg_1_tensor.clone()
arg_2 = 2239310108
arg_3 = 2
try:
  res = torch.tensor_split(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))