import torch.nn as nn

import torch

class SparseTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("a", torch.eye(3).to_sparse())

    def forward(self):
        pass

s = SparseTensorModule()
s.a
print("hi")
print(s.state_dict())

hi
OrderedDict([('a', tensor(indices=tensor([[0, 1, 2],
                       [0, 1, 2]]),
       values=tensor([1., 1., 1.]),
       size=(3, 3), nnz=3, layout=torch.sparse_coo))])

self.register_buffer("a", torch.eye(3).to_sparse())

self.register_buffer("a", torch.eye(3).to_sparse_csr())