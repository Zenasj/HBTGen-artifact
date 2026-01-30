import torch
arg_1_tensor = torch.neg(torch.rand([2, 3, 1], dtype=torch.float32))
arg_1 = arg_1_tensor.clone()
arg_2_tensor = torch.rand([2, 3, 3], dtype=torch.float32)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = torch.randint(-2048,8,[2, 3], dtype=torch.int32)
arg_3 = arg_3_tensor.clone()
try:
  results["res_cpu"] = torch.lu_solve(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))
arg_1 = arg_1_tensor.clone().cuda()
arg_2 = arg_2_tensor.clone().cuda()
arg_3 = arg_3_tensor.clone().cuda()
try:
  results["res_gpu"] = torch.lu_solve(arg_1,arg_2,arg_3,)
except Exception as e:
  print("Error:"+str(e))