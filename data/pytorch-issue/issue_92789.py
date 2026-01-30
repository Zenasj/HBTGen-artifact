results = dict()
import torch
arg_1_tensor = torch.randint(-512,512,[100, 100, 100, 5, 5, 5], dtype=torch.int64)
arg_1 = arg_1_tensor.clone()
try:
  results["res_cpu"] = torch.Tensor.indices(arg_1,)
except Exception as e:
  print("Error:"+str(e))
arg_1 = arg_1_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.Tensor.indices(arg_1,)
except Exception as e:
  print("Error:"+str(e))

print(results)