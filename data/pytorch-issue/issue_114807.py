import torch

X = torch.randn(2, 2).to_sparse_csc() 
Y = torch.randn(2, 2)
X + Y