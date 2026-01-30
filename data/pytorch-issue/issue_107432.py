import torch.nn as nn

results = dict()
import torch
arg_1_tensor = torch.rand([10, 3], dtype=torch.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-128,64,[8], dtype=torch.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.tensor([1], dtype=torch.bool)
arg_3 = arg_3_tensor.clone()
arg_4 = True
try:
  results["res_cpu"] = torch.nn.functional.embedding_bag(arg_1,arg_2,arg_3,sparse=arg_4,)
except Exception as e:
  print("Error:"+str(e))
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.embedding_bag(arg_1,arg_2,arg_3,sparse=arg_4,)
except Exception as e:
  print("Error:"+str(e))

print(results)

tensor([[0.0217, 0.1224, 0.0373]], device='cuda:0', dtype=torch.float64)