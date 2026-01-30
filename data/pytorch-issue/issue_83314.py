results = dict()
import torch
arg_1 = torch.randint(-32768,8,[2], dtype=torch.int64)
arg_2 = torch.randint(-1,32,[4], dtype=torch.int8)
try:
  results["res_cpu"] = torch.Tensor.equal(arg_1,arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1.clone().cuda()
arg_2 = arg_2.clone().cuda()
try:
  results["res_gpu"] = torch.Tensor.equal(arg_1,arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
print(results)