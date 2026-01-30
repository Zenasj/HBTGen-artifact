import torch
from torch.sparse import to_sparse_semi_structured

a = torch.randn(3072, 768).cuda().half()/100
pruning_inds = a.abs().view(-1, 4).argsort(dim=1)[:, :2]

a.view(-1, 4).scatter_(1, pruning_inds, torch.zeros_like(pruning_inds).half())

b = to_sparse_semi_structured(a).to_dense()

print(((a-b)**2).sum())
print(a[a!=b][:32])
print(b[a!=b][:32])

tensor(239.8750, device='cuda:0', dtype=torch.float16)
tensor([ 0.0000, -0.0087,  0.0000,  0.0162,  0.0000, -0.0073,  0.0163,  0.0000,                                                                                                                                                       
         0.0000,  0.0155,  0.0000, -0.0177,  0.0128,  0.0000,  0.0000, -0.0084,                                                                                                                                                       
        -0.0058,  0.0000,  0.0000,  0.0124,  0.0000,  0.0093,  0.0000,  0.0060,                                                                                                                                                       
         0.0078,  0.0145,  0.0000, -0.0056,  0.0000,  0.0157,  0.0146,  0.0000],                                                                                                                                                      
       device='cuda:0', dtype=torch.float16)
tensor([-0.0087,  0.0000,  0.0162,  0.0000, -0.0073,  0.0000,  0.0000,  0.0163,                                                                                                                                                       
         0.0155,  0.0000, -0.0177,  0.0000,  0.0000,  0.0128, -0.0084,  0.0000,                                                                                                                                                       
         0.0000, -0.0058,  0.0124,  0.0000,  0.0093,  0.0000,  0.0060,  0.0000,
         0.0000,  0.0078,  0.0145,  0.0000, -0.0056,  0.0000,  0.0157,  0.0146],
       device='cuda:0', dtype=torch.float16)

from torch.sparse import to_sparse_semi_structured

from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
SparseSemiStructuredTensor._FORCE_CUTLASS = True

import torch
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor

def test(force_cutlass):
    print(f"TESTING CUTLASS: {force_cutlass}")
    print("-------------------")
    SparseSemiStructuredTensor._FORCE_CUTLASS = force_cutlass
    torch.manual_seed(52)
    a = torch.randn(256, 256).cuda().half()/100

    pruning_inds = a.abs().view(-1, 4).argsort(dim=1)[:, :2]

    a.view(-1, 4).scatter_(1, pruning_inds, torch.zeros_like(pruning_inds).half())
    a_sparse = to_sparse_semi_structured(a)
    print(a_sparse.indices())
    b = a_sparse.to_dense()

    print(((a-b)**2).sum())
    print(a[a!=b][:32])
    print(b[a!=b][:32])

    return a_sparse.indices().detach()

cutlass_indicies = test(True)
cusparselt_indicies = test(False)

torch.set_printoptions(profile="full")
print("different indices at locations:")
print(torch.nonzero((cutlass_indicies - cusparselt_indicies)**2))

from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
SparseSemiStructuredTensor._FUSE_TRANSPOSE = True