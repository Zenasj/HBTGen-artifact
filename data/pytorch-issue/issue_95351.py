import torch

sparse_tensor = torch.sparse_coo_tensor([[1,1]], [1,1], (2,))
torch.Tensor.clamp(sparse_tensor, -1, 1)
# torch.Tensor.clamp_(sparse_tensor, -1, 1)