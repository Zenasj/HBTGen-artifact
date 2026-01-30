import torch

torch.autograd.gradcheck(lambda x: torch.Tensor.t(x).to_dense(masked=False), a, masked=False)

torch.autograd.gradcheck(lambda x: torch.Tensor.t(x).to_dense(masked=True), a, masked=True)