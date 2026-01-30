import torch

print(torch.__version__)

input = torch.randn(0)
res = torch.ops.aten._local_scalar_dense(input)

print(res)