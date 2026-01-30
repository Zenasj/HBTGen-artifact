import torch.nn as nn

import torch
arg_1_tensor = -torch.rand([1, 64, 10, 9, 8], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = False
try:
  results["res_cpu"] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))

{'res_cpu': tensor([], size=(1, 64, 0, 0, 0)), 'res_gpu': tensor([], device='cuda:0', size=(1, 64, 0, 0, 0))}