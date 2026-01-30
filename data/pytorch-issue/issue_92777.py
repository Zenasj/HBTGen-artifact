import torch.nn as nn

results = dict()
import torch
arg_1_tensor = torch.rand([2, 3, 6, 4, 10], dtype=torch.float64)
arg_1 = arg_1_tensor.clone()
arg_2_0 = 0
arg_2_1 = 0
arg_2_2 = 0
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
arg_3 = False
try:
  results["res_cpu"] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = [arg_2_0,arg_2_1,arg_2_2,]
try:
  results["res_gpu"] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))

print(results)

results = dict()
import torch
arg_1_tensor = torch.rand([1, 1, 3, 3, 3], dtype=torch.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = 0
try:
  results["res_cpu"] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,)
except Exception as e:
  print("Error:"+str(e))
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.adaptive_max_pool3d(arg_1,arg_2,)
except Exception as e:
  print("Error:"+str(e))

print(results)