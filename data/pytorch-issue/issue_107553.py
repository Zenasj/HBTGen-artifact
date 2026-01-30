import torch.nn as nn

import torch
import numpy as np
arg_1 = 1250999896765
arg_2 = "bilinear"
arg_3 = False
arg_class = torch.nn.Upsample(scale_factor=arg_1,mode=arg_2,align_corners=arg_3,)
arg_4_0_tensor = torch.rand([5, 3, 64, 64], dtype=torch.float32)
arg_4_0 = arg_4_0_tensor.clone()
arg_4 = [arg_4_0,]
try:
  res = arg_class(*arg_4)
except Exception as e:
  print("Error:"+str(e))