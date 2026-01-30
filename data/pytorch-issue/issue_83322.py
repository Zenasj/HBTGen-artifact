import torch.nn as nn

results = dict()
import torch
arg_1 = torch.rand([3], dtype=torch.float16)
arg_2 = torch.rand([3], dtype=torch.float32)
arg_3 = None
arg_4 = None
arg_5 = "mean"
try:
  results["res_cpu"] = torch.nn.functional.binary_cross_entropy_with_logits(arg_1,arg_2,arg_3,pos_weight=arg_4,reduction=arg_5,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1.clone().cuda()
arg_2 = arg_2.clone().cuda()
try:
  results["res_gpu"] = torch.nn.functional.binary_cross_entropy_with_logits(arg_1,arg_2,arg_3,pos_weight=arg_4,reduction=arg_5,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
print(results)