import torch

# self.weight is sparsed with coo format
x = torch.sparse.mm(self.weight, x)