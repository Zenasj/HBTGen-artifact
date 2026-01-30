results = dict()
import torch
arg_1_tensor = torch.rand([5, 5], dtype=torch.float64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.randint(-16384,2048,[5], dtype=torch.int32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.neg(torch.rand([5, 1], dtype=torch.float64))
arg_3 = arg_3_tensor.clone()
arg_4 = False
try:
  results["res_cpu"] = torch.linalg.ldl_solve(arg_1,arg_2,arg_3,hermitian=arg_4,)
except Exception as e:
  print("Error:"+str(e))
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.linalg.ldl_solve(arg_1,arg_2,arg_3,hermitian=arg_4,)
except Exception as e:
  print("Error:"+str(e))

print(results)