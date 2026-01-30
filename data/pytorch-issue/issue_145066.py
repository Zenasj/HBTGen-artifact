import torch

print(torch.__version__)

input = torch.randn(size=(0,))
torch.ops.aten._local_scalar_dense(input)