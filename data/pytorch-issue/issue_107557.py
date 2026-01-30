import torch.nn as nn

import torch
arg_1_0 = 0
arg_1_1 = None
arg_1_2 = None
arg_1 = [arg_1_0,arg_1_1,arg_1_2,]
arg_class = torch.nn.AdaptiveMaxPool3d(arg_1,)
arg_2_0_tensor = torch.rand([1, 64, 10, 9, 8], dtype=torch.float32)
arg_2_0 = arg_2_0_tensor.clone()
arg_2 = [arg_2_0,]
try:
  results["res_cpu"] = arg_class(*arg_2)
except Exception as e:
  print("Error:"+str(e))
arg_class = arg_class.cuda()
arg_2_0 = arg_2_0_tensor.clone().cuda()
arg_2 = [arg_2_0,]
try:
  results["res_gpu"] = arg_class(*arg_2)
except Exception as e:
  print("Error:"+str(e))

{'res_cpu': tensor([], size=(1, 64, 0, 9, 8)), 'res_gpu': tensor([], device='cuda:0', size=(1, 64, 0, 9, 8))}