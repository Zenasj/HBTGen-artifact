import torch
import numpy as np
arg_1_tensor = torch.neg(torch.rand([100, 100, 100, 5, 5, 5], dtype=torch.complex128))
arg_1 = arg_1_tensor.clone()
try:
  res = torch.Tensor.is_coalesced(arg_1,)
except Exception as e:
  print("Error:"+str(e))