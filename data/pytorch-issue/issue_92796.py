import torch.nn as nn

results = dict()
import torch
arg_1_tensor = torch.rand([2, 2, 4, 4, 4], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
arg_3 = True
try:
  results["res_cpu"] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))

print(results)

results = dict()
import torch
arg_1_tensor = torch.rand([3, 5, 6, 7], dtype=torch.float32)
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

print(results)