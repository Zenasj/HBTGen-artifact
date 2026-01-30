import torch

print(torch.__version__)

var_199 = torch.rand((0,))
torch.ops.aten._local_scalar_dense(var_199)