results = dict()
import torch
arg_1 = torch.rand([2, 3, 1], dtype=torch.float32)
arg_2 = torch.rand([2, 3, 3], dtype=torch.float32)
arg_3 = torch.randint(-10,10,[2, 3], dtype=torch.int32)
try:
  results["res_cpu"] = torch.lu_solve(arg_1,arg_2,arg_3,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_4 = arg_1.clone().cuda()
arg_5 = arg_2.clone().cuda()
arg_6 = arg_3.clone().cuda()
try:
  results["res_gpu"] = torch.lu_solve(arg_4,arg_5,arg_6,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
print(results)