import torch

torch.addmm(bias, dense, sparse.t())
torch.mm(dense, sparse)
torch.mm(sparse, dense)
aten.linear.default
aten.t.default
aten.t.detach

with torch.sparse.check_sparse_tensor_invariants():
    model_incorrectly_using_semi_structured_sparse()

model_incorrectly_using_semi_structured_sparse()