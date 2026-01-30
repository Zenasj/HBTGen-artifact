import torch
import numpy as np
arg_1_tensor = torch.randint(-16,4,[], dtype=torch.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(0,1,[1], dtype=torch.uint8)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.neg(torch.rand([3, 3, 3], dtype=torch.complex128))
arg_3 = arg_3_tensor.clone()
try:
  res = torch.eq(arg_1,arg_2,out=arg_3,)
except Exception as e:
  print("Error:"+str(e))