import torch.nn as nn

import torch
import numpy as np
arg_1_tensor = torch.randint(0,256,[1, 1, 1, 0], dtype=torch.uint8)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
try:
  res = torch.nn.functional.pixel_shuffle(arg_1,upscale_factor=arg_2,)
except Exception as e:
  print("Error:"+str(e))

import torch
import numpy as np
arg_1_tensor = torch.randint(0,2,[1, 0, 1, 1], dtype=torch.bool)
arg_1 = arg_1_tensor.clone()
arg_2 = 1
try:
  res = torch.nn.functional.pixel_shuffle(arg_1,upscale_factor=arg_2,)
except Exception as e:
  print("Error:"+str(e))