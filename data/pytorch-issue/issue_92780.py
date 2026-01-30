import torch
import numpy as np
arg_1_tensor = torch.neg(torch.rand([5], dtype=torch.float32))
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.neg(torch.rand([5, 0], dtype=torch.float16))
arg_2 = arg_2_tensor.clone()
try:
  res = torch.Tensor.copy_(arg_1,arg_2,)
except Exception as e:
  print("Error:"+str(e))