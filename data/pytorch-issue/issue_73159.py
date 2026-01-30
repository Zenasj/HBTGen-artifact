results = dict()
import torch
input = torch.rand([0, 5, 5], dtype=torch.float64)
dim = 33
try:
  results["res_cpu"] = torch.logcumsumexp(input, dim)
except Exception as e:
  results["err_cpu"] = str(e)
input_cuda = input.cuda()
try:
  results["res_gpu"] = torch.logcumsumexp(input_cuda, dim)
except Exception as e:
  results["err_gpu"] = str(e)

print(results)
# {'err_cpu': 'Dimension out of range (expected to be in range of [-3, 2], but got 33)', 'res_gpu': tensor([], device='cuda:0', size=(0, 5, 5), dtype=torch.float64)}