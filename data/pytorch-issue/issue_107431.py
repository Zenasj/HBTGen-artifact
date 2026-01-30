import torch
import numpy as np
arg_1_tensor = torch.randint(-1,8,[4], dtype=torch.int8)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-16,16,[], dtype=torch.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.randint(414,458,[3, 3, 3], dtype=torch.float32)
arg_3 = arg_3_tensor.clone()
try:
  res = torch.logical_and(arg_1,arg_2,out=arg_3,)
except Exception as e:
  print("Error:"+str(e))