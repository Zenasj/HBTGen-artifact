import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.randint(-2048,32768,[1, 1, 1, 0], dtype=torch.int32)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
try:
  res = torch.nn.functional.pixel_unshuffle(arg_1,downscale_factor=arg_2,)
except Exception as e:
  print("Error:"+str(e))

import torch
import numpy as np
arg_1_tensor = torch.randint(-128,16,[1, 1, 1, 0], dtype=torch.int64)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
try:
  res = torch.nn.functional.pixel_unshuffle(arg_1,downscale_factor=arg_2,)
except Exception as e:
  print("Error:"+str(e))