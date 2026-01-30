import torch
results={}
arg_1 = torch.as_tensor([[-106, -190, -960,  -28, -379, -176, -893, -587, -352, -136]])
arg_2 = 0
try:
  results["res_cpu"] = torch.floor_divide(input=arg_1,other=arg_2,)
except Exception as e:
  results["err_cpu"] = "ERROR:"+str(e)
arg_1 = arg_1.clone().cuda()
try:
  results["res_gpu"] = torch.floor_divide(input=arg_1,other=arg_2,)
except Exception as e:
  results["err_gpu"] = "ERROR:"+str(e)
print(results)
# results={'err_cpu': 'ERROR:ZeroDivisionError', 'res_gpu': tensor([[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2]], device='cuda:0')}