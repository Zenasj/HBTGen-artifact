import torch

candidate = torch.tensor([[[-1, -2], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
[[-11, -12], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]],
[[-101, -102], [100, 101], [102, 103], [104, 105], [106, 107], [108, 109]]])
mask = torch.tensor([[0, 1, 1, 1, 1, 0],
[0, 1, 1, 0, 1, 1],
[0, 1, 1, 0, 1, 1]], dtype=torch.bool)
result_cpu = torch.masked_select(candidate.permute(2, 0, 1), mask)
result_gpu = torch.masked_select(candidate.to('cuda').permute(2, 0, 1), mask.to('cuda'))

assert not (result_cpu != result_gpu.cpu()).any()

x = torch.randn(3, 3)
mask = torch.ones(3, 3, dtype=torch.bool)
cpu_output = torch.masked_select(x.t(), mask)
cuda_output = torch.masked_select(x.cuda().t(), mask.cuda())
assert torch.allclose(cpu_output, cuda_output.cpu())