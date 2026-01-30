import torch
import torch.optim as optim

features = 3

param = torch.zeros(features, layout=torch.sparse_coo, requires_grad=True)
param.grad = torch.sparse_coo_tensor([[0]], 1., size=(features,))
optimizer = optim.SparseAdam([param])

optimizer.step()
# RuntimeError: Cannot access data pointer of Tensor that doesn't have storage