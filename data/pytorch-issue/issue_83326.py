results = dict()
import torch
arg_1 = torch.rand([4], dtype=torch.float32)
try:
  results["res_cpu"] = torch.tril(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = arg_1.clone().cuda()
try:
  results["res_gpu"] = torch.tril(arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
print(results)

results = dict()
import torch
arg_1 = torch.rand([], dtype=torch.float32)
try:
  results["res_cpu"] = torch.triu(arg_1,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_2 = arg_1.clone().cuda()
try:
  results["res_gpu"] = torch.triu(arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
print(results)