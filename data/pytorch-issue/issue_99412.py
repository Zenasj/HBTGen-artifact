import torch.nn as nn

import torch
results={}
arg_1 = 3
arg_2 = 255
arg_3 = False
arg_class = torch.nn.MaxPool1d(kernel_size=arg_1,stride=arg_2,return_indices=arg_3,)
arg_4_0 = torch.as_tensor([[0.3204]])
print(arg_4_0)
arg_4 = [arg_4_0,]
try:
  results["res_cpu"] = arg_class(*arg_4)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_class = arg_class.cuda()
arg_4_0 = arg_4_0.clone().cuda()
arg_4 = [arg_4_0,]
try:
  results["res_gpu"] = arg_class(*arg_4)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
print(results)
# results: {'res_cpu': tensor([], size=(1, 0)), 'err_gpu': 'ERROR:Given input size: (1x1x1). Calculated output size: (1x1x0). Output size is too small'}