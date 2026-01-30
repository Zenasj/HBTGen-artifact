import torch

torch.ones(3, 1)*torch.ones(3, 10)

i = torch.tensor([[0, 1, 1],
                          [2, 0, 2]])
a = torch.sparse_coo_tensor(i, torch.ones(3, 1), [2, 4, 1])
b = torch.sparse_coo_tensor(i, torch.ones(3, 10), [2, 4, 10])
# gives RuntimeError: mul operands have incompatible sizes