import torch
import numpy as np
arg_1_tensor = torch.randint(-8192,128,[3], dtype=torch.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 433894953
try:
  res = torch.Tensor.repeat_interleave(arg_1,arg_2,)
except Exception as e:
  print("Error:"+str(e))