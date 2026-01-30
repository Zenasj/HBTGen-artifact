import torch
from torch.testing._internal.common_utils import TestCase
t = TestCase()

size = (10, 10)
for size in [(10, 10), (15, 20)]:
    for nnz in [10, 40]:
        a = t.genSparseCSRTensor(size, nnz,
                                 device='cpu',
                                 dtype=torch.float32,
                                 index_dtype=torch.int32)
        predicted = torch.ones(size[0]+1, dtype=torch.int32) * nnz
        r = torch.arange(0, nnz, max(nnz // size[0], 1), dtype=torch.int32)[:size[0]+1]
        predicted[:r.numel()] = r
        assert (a.crow_indices() == predicted).all()
        print(predicted)