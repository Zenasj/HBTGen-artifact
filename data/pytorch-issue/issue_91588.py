import torch
import numpy as np
arg_1_tensor = torch.rand([2], dtype=torch.bfloat16)
arg_1 = arg_1_tensor.clone()
arg_2 = 102919646178
arg_3 = False
try:
  res = torch.combinations(arg_1,r=arg_2,with_replacement=arg_3,)
except Exception as e:
  print("Error:"+str(e))