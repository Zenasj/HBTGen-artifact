import torch
a = torch.rand([0, 4])
dim = 0
indices = torch.tensor([0, 1])
torch.index_select(a, dim, indices)
# tensor([[1.1704e-19, 1.3563e-19, 7.7098e-33, 1.3594e-19],
#       [1.3563e-19, 2.6451e+20, 1.2708e+31, 6.1186e-04]])