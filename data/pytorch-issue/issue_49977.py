import torch

# raises exception
torch.tensor([True, False]).to_sparse().to_dense()
# RuntimeError

# converting via another type works
torch.tensor([True, False]).to_sparse().to(torch.uint8).to_dense().to(torch.bool)
# Out[6]: tensor([ True, False])

x = torch.tensor([[[True, False],[True, True],[True, True]]])
x_sparse = x.to_sparse()
x_dense = x_sparse.to_dense()